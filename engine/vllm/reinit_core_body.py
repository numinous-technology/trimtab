    def trimtab_reinit(self, fields: dict) -> dict:
        """trimtab warm reinit. Rebuild the KV cache, attention groups, CUDA
        graphs and the scheduler at a new size. Weights never leave the GPU.

        Accepts gpu_memory_utilization, max_num_seqs, max_num_batched_tokens.
        Needs an idle engine (callers drain) and --enable-sleep-mode.
        """
        import time as _time

        t0 = _time.perf_counter()
        allowed = {"gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens"}
        bad = sorted(set(fields) - allowed)
        if bad:
            return {"ok": False, "error": f"not warm-reinitable: {bad}"}
        if not self.vllm_config.model_config.enable_sleep_mode:
            return {"ok": False, "error": "warm reinit needs the server launched with --enable-sleep-mode"}
        if self.scheduler.has_requests():
            return {"ok": False, "error": "engine busy, drain before a warm reinit"}

        cc, sc = self.vllm_config.cache_config, self.vllm_config.scheduler_config
        previous = {"gpu_memory_utilization": cc.gpu_memory_utilization,
                    "max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens}
        released = self.collective_rpc("trimtab_release_kv")
        freed_at = _time.perf_counter()

        def _rebuild(values):
            if "gpu_memory_utilization" in values:
                cc.gpu_memory_utilization = float(values["gpu_memory_utilization"])
            if "max_num_seqs" in values:
                sc.max_num_seqs = int(values["max_num_seqs"])
            if "max_num_batched_tokens" in values:
                sc.max_num_batched_tokens = int(values["max_num_batched_tokens"])
            cc.num_gpu_blocks = None
            kv_cache_config = self._initialize_kv_caches(self.vllm_config)
            block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, self.vllm_config)
            old = self.scheduler
            self.scheduler = type(old)(
                vllm_config=self.vllm_config,
                kv_cache_config=kv_cache_config,
                structured_output_manager=self.structured_output_manager,
                include_finished_set=old.finished_req_ids_dict is not None,
                log_stats=self.log_stats,
                block_size=block_size,
                hash_block_size=hash_block_size,
            )
            return kv_cache_config

        try:
            kv_cache_config = _rebuild(fields)
        except Exception as e:
            logger.warning("trimtab reinit with %s failed (%s), restoring previous sizing", fields, e)
            self.collective_rpc("trimtab_release_kv")
            kv_cache_config = _rebuild(previous)
            fields = {"error": str(e), "restored": previous}

        sched = self.scheduler
        sched._trimtab_ceilings = {"max_num_seqs": sched.max_num_running_reqs,
                                   "max_num_batched_tokens": sched.max_num_scheduled_tokens}
        done = _time.perf_counter()
        info = {
            "ok": "error" not in fields, "fields": fields,
            "num_gpu_blocks": kv_cache_config.num_blocks,
            "max_num_seqs": sched.max_num_running_reqs,
            "freed_gib": [r.get("freed_gib") for r in released],
            "free_s": round(freed_at - t0, 3), "rebuild_s": round(done - freed_at, 3), "total_s": round(done - t0, 3),
        }
        self._trimtab_last_reinit = info
        logger.info("trimtab warm reinit %s", info)
        return info

