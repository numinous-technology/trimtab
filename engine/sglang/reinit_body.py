    def trimtab_reinit(self, fields: dict) -> dict:
        """trimtab warm reinit. Rebuild KV pools, attention backends and CUDA
        graphs in place with new sizing. Weights never leave the GPU.

        Accepts mem_fraction_static, max_total_tokens, kv_cache_dtype,
        max_running_requests. Needs an idle scheduler, callers drain first.
        Runs on every rank through the same fan-out as set_internal_state.
        """
        import gc
        import time as _time

        t0 = _time.perf_counter()
        allowed = {"mem_fraction_static", "max_total_tokens", "kv_cache_dtype", "max_running_requests"}
        bad = sorted(set(fields) - allowed)
        if bad:
            return {"ok": False, "error": f"not warm-reinitable: {bad}"}
        _saver = getattr(self.tp_worker.model_runner, "memory_saver_adapter", None)
        if _saver is None or "Noop" in type(_saver).__name__:
            info = {"ok": False, "error": "warm reinit needs the server launched with --enable-memory-saver"}
            self._trimtab_last_reinit = info
            return info
        pending = getattr(self, "result_queue", None)
        if not self.is_fully_idle() or (pending is not None and len(pending)) or self.last_batch is not None:
            info = {"ok": False, "error": "scheduler busy, drain before a warm reinit"}
            self._trimtab_last_reinit = info
            return info

        self.flush_cache(empty_cache=False)
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.last_batch = None
        self.chunked_req = None

        mr = self.tp_worker.model_runner
        previous = {
            "mem_fraction_static": mr.mem_fraction_static,
            "max_total_tokens": self.max_total_num_tokens,
            "max_running_requests": self.max_running_requests,
        }
        kinds = ("token_to_kv_pool", "token_to_kv_pool_allocator", "req_to_token_pool", "_unified_memory_pool",
                 "decode_cuda_graph_runner", "prefill_cuda_graph_runner", "tree_cache")
        old = {k: (self.tree_cache if k == "tree_cache" else getattr(mr, k, None)) for k in kinds}
        old = {k: v for k, v in old.items() if v is not None}
        free_before = torch.cuda.mem_get_info()[0]

        workers = [w for w in (self.tp_worker, getattr(self, "draft_worker", None)) if w is not None]
        for w in workers:
            r = w.model_runner
            for attr in ("decode_cuda_graph_runner", "prefill_cuda_graph_runner", "eager_runner",
                         "token_to_kv_pool", "token_to_kv_pool_allocator", "req_to_token_pool",
                         "_unified_memory_pool", "attn_backend", "decode_attn_backend", "prefill_attn_backend",
                         "graph_shared_output"):
                if hasattr(r, attr):
                    setattr(r, attr, None)
            if "mem_fraction_static" in fields:
                r.mem_fraction_static = float(fields["mem_fraction_static"])
        self.tree_cache = None
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        released, slots = [], []
        def _hollow(o, depth=0):
            """Drop every tensor and CUDA graph the object owns, one level deep,
            so its GPU memory is released even while stale references to the
            object itself survive elsewhere."""
            d = getattr(o, "__dict__", None)
            if d is None:
                return
            for k, v in list(d.items()):
                if torch.is_tensor(v) or type(v).__name__ == "CUDAGraph":
                    d[k] = None
                elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) or type(x).__name__ == "CUDAGraph" for x in v):
                    d[k] = type(v)()
                elif isinstance(v, dict) and v and all(torch.is_tensor(x) or type(x).__name__ == "CUDAGraph" for x in v.values()):
                    d[k] = {}
                elif depth == 0 and hasattr(v, "__dict__") and not isinstance(v, type):
                    _hollow(v, 1)
        saver = getattr(mr, "memory_saver_adapter", None)
        saver_on = saver is not None and "Noop" not in type(saver).__name__
        gen = getattr(self, "_trimtab_gen", 0)
        if saver_on:
            # --enable-memory-saver: unmap the physical pages of every allocation
            # tagged kv_cache and cuda_graph, whatever still references them.
            # Each reinit generation allocates under its own tag suffix so a
            # later pause never touches an already paused generation.
            suffix = f".g{gen}" if gen else ""
            for tag in ("kv_cache" + suffix, "cuda_graph" + suffix):
                try:
                    saver.pause(tag)
                except Exception as e:
                    logger.warning(f"trimtab reinit pause({tag}) skipped: {e}")
            gen += 1
            self._trimtab_gen = gen
            # Pool classes create their own adapter instances, so the tag
            # mapping lives on the class and follows the current generation.
            cls = type(saver)
            if not hasattr(cls, "_trimtab_real_region"):
                cls._trimtab_real_region = cls.region

                def _region(inst, tag, enable_cpu_backup=False):
                    g = getattr(cls, "_trimtab_gen", 0)
                    if g and tag in ("kv_cache", "cuda_graph"):
                        tag = f"{tag}.g{g}"
                    return cls._trimtab_real_region(inst, tag, enable_cpu_backup)

                cls.region = _region
            cls._trimtab_gen = gen
        else:
            for obj in old.values():
                try:
                    _hollow(obj)
                except Exception as e:
                    logger.warning(f"trimtab reinit hollow skipped: {e}")
        # With the saver on, the old tensors stay alive on purpose. Their
        # physical pages are unmapped, and keeping them referenced stops the
        # caching allocator from handing their (now unmapped) segments to the
        # new pools. Only virtual address space is retained per generation.
        if saver_on:
            self._trimtab_paused = getattr(self, "_trimtab_paused", []) + [old]
        for attr in ("kv_cache_configurator", "canary_manager", "kv_index_translator"):
            if hasattr(mr, attr):
                setattr(mr, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_after = torch.cuda.mem_get_info()[0]
        freed_at = _time.perf_counter()
        logger.info(f"trimtab reinit released {sorted(set(released))}, freed {(free_after - free_before) / 2**30:.2f} GiB")

        def _rebuild(values):
            get_context().override(source="trimtab_reinit", **values)
            for w in workers:
                if "mem_fraction_static" in values:
                    w.model_runner.mem_fraction_static = float(values["mem_fraction_static"])
            self.init_memory_pools()
            self.init_all_attention_backends()
            self.init_all_cuda_graphs()

        try:
            _rebuild(fields)
        except Exception as e:  # rebuild with the previous sizing so the server survives
            logger.warning(f"trimtab reinit with {fields} failed ({e}), restoring previous sizing")
            _rebuild({k: previous[k] for k in previous if k in fields or k == "mem_fraction_static"})
            fields = {"error": str(e), "restored": previous}

        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            _, _, _, _, _, _,
        ) = self.tp_worker.get_worker_info()
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            enable_metrics=get_observability().enable_metrics,
            enable_kv_cache_events=False,
            ps=self.ps,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            hicache_draft_plan=(self.draft_worker.hicache_draft_plan if getattr(self, "draft_worker", None) is not None else None),
        )
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.tree_cache = result.tree_cache
        new = {k: (self.tree_cache if k == "tree_cache" else getattr(mr, k, None)) for k in kinds}
        # Scheduler components (frozen slotted dataclasses, SchedulePolicy, workers)
        # were built with the old pools and tree. Rewire every field that still
        # points at an old object to the rebuilt one of the same kind.
        rewired = []
        components = list(vars(self).values()) + [self.tp_worker, mr] + ([self.draft_worker] if getattr(self, "draft_worker", None) else [])
        for comp in components:
            if comp is None or isinstance(comp, (int, float, str, bytes, list, dict, tuple, set, type)):
                continue
            for field in ("tree_cache", "token_to_kv_pool_allocator", "req_to_token_pool", "token_to_kv_pool"):
                try:
                    cur = getattr(comp, field)
                except Exception:
                    continue
                target = new.get(field)
                if target is None or cur is target or cur is None:
                    continue
                if type(cur) is type(target) or any(cur is o for o in old.values()):
                    object.__setattr__(comp, field, target)
                    rewired.append(f"{type(comp).__name__}.{field}")
        self.new_token_ratio_tracker.reset()
        self._trimtab_ceilings = {"max_running_requests": self.max_running_requests}
        done = _time.perf_counter()
        info = {
            "ok": "error" not in fields, "fields": fields,
            "released": sorted(set(released)), "freed_gib": round((free_after - free_before) / 2**30, 2),
            "memory_saver": saver_on, "rewired": sorted(set(rewired)),
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_running_requests": self.max_running_requests,
            "free_s": round(freed_at - t0, 3), "rebuild_s": round(done - freed_at, 3), "total_s": round(done - t0, 3),
        }
        self._trimtab_last_reinit = info
        logger.info(f"trimtab warm reinit {info}")
        return info

