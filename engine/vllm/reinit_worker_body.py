    def trimtab_release_kv(self) -> dict:
        """trimtab warm reinit, worker side. Free the KV pool and drop the
        captured CUDA graphs. Weights stay on the GPU.

        The KV cache is allocated inside CuMemAllocator.use_memory_pool(
        tag="kv_cache"). Dropping the tensor references marks the blocks free,
        but torch.cuda.empty_cache() errors on a pluggable allocator
        (pytorch/pytorch#145168), so the pages are not returned. vLLM's own
        use_memory_pool exit works around this by snapshotting the pool and
        releasing every allocated_size==0 block by hand. We run the same
        release here, which returns the physical pages cleanly (single unmap,
        popped from pointer_to_data) with no double-unmap. discard() is the
        wrong primitive: it marks blocks asleep and later double-unmaps.
        """
        import gc

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release

        mr = self.model_runner
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info()[0]

        try:
            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

            BreakableCUDAGraphWrapper.clear_all_graphs()
        except Exception:
            pass
        CUDAGraphWrapper.clear_all_graphs()
        for key_set in mr.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        mr.cudagraph_dispatcher.keys_initialized = False

        # Drop every KV reference (attention KV and hybrid mamba state both live
        # in mr.kv_caches and layer.kv_cache, which this clears).
        mr._cleanup_profiling_kv_cache()
        gc.collect()

        # Return the freed blocks to the OS, the way use_memory_pool exit does.
        allocator = CuMemAllocator.get_instance()
        released = 0
        data = allocator.allocator_and_pools.get("kv_cache")
        if data is not None:
            for allocation in data[0].snapshot():
                if allocation["allocated_size"] == 0:
                    handle = allocator._python_free_callback(allocation["address"])
                    unmap_and_release(handle)
                    released += handle[1]
        torch.cuda.synchronize()

        # vLLM caps the process at gpu_memory_utilization through
        # set_per_process_memory_fraction. Lift it, the profiler sizes the new
        # pool against device free memory, not this guard.
        torch.cuda.set_per_process_memory_fraction(1.0)

        # Refresh the profiler baseline, captured at boot, to the current state:
        # weights resident, KV freed. MemorySnapshot measures on construction.
        from vllm.utils.mem_utils import MemorySnapshot

        self.init_snapshot = MemorySnapshot(device=self.device)
        return {"released_gib": round(released / 2**30, 2),
                "free_gib": round((torch.cuda.mem_get_info()[0] - free0) / 2**30, 2)}

