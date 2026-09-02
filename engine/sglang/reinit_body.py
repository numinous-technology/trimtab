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
        if not self.is_fully_idle():
            return {"ok": False, "error": "scheduler busy, drain before a warm reinit"}

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
        # Anything else still pointing at the old pools keeps their GPU memory
        # alive. Release it generically and log the owner, so the explicit list
        # above can grow from evidence rather than guesswork.
        released, slots = [], []
        for kind, obj in old.items():
            for ref in gc.get_referrers(obj):
                if isinstance(ref, dict):
                    for k, v in list(ref.items()):
                        if v is obj:
                            ref[k] = None
                            released.append(f"{kind}<-{k}")
                            slots.append((ref, k, kind))
                elif isinstance(ref, list):
                    for i, v in enumerate(ref):
                        if v is obj:
                            ref[i] = None
                            released.append(f"{kind}<-list")
                            slots.append((ref, i, kind))
        del old, obj
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
        for ref, key, kind in slots:  # rewire every released slot to the rebuilt object of the same kind
            try:
                if ref[key] is None:
                    ref[key] = new[kind]
            except (KeyError, IndexError, TypeError):
                pass
        self.new_token_ratio_tracker.reset()
        self._trimtab_ceilings = {"max_running_requests": self.max_running_requests}
        done = _time.perf_counter()
        info = {
            "ok": "error" not in fields, "fields": fields,
            "released": sorted(set(released)), "freed_gib": round((free_after - free_before) / 2**30, 2),
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_running_requests": self.max_running_requests,
            "free_s": round(freed_at - t0, 3), "rebuild_s": round(done - freed_at, 3), "total_s": round(done - t0, 3),
        }
        self._trimtab_last_reinit = info
        logger.info(f"trimtab warm reinit {info}")
        return info

