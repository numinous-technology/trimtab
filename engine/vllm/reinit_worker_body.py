    def trimtab_release_kv(self) -> dict:
        """trimtab warm reinit, worker side. Unmap the KV pool and drop the
        captured CUDA graphs. Weights stay on the GPU.

        Needs --enable-sleep-mode so the KV pool lives in the CuMemAllocator,
        whose discard() unmaps a tag's physical pages regardless of who still
        references the tensors. The old tensors are kept alive on purpose so
        the allocator's pool never hands their unmapped segments to the new KV
        cache. Only virtual address space is retained per reinit.
        """
        import gc

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.device_allocator.cumem import CuMemAllocator

        mr = self.model_runner
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info()[0]
        self._trimtab_old_kv = getattr(self, "_trimtab_old_kv", []) + [list(mr.kv_caches)]
        try:
            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

            BreakableCUDAGraphWrapper.clear_all_graphs()
        except Exception:
            pass
        CUDAGraphWrapper.clear_all_graphs()
        for key_set in mr.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        mr.cudagraph_dispatcher.keys_initialized = False
        mr._cleanup_profiling_kv_cache()
        CuMemAllocator.get_instance().discard("kv_cache")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return {"freed_gib": round((torch.cuda.mem_get_info()[0] - free0) / 2**30, 2)}

