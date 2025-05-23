class DeepSpeedConfig:
    def __init__(self):
        self.datastates_config = self.DataStatesConfig()
    
    class DataStatesConfig:
        def __init__(self):
            self.enabled = True
            self.config = {
                "host_cache_size": 1,
                "parser_threads": 2,
                "pin_host_cache": True
            }

def test_deepspeed():
    deepspeed_config = DeepSpeedConfig()
    print(f"Going to initalize datastates engine...")
    ckpt_engine = Checkpointing(runtime_config=deepspeed_config, rank=0)
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    tensor_shape = torch.Size([256, 256])
    tensor_dtype = torch.float32
    tensor = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    
    model_name = ""
    np_array = np.random.randn(512).astype(np.float32)
    ckpt_path = "/checkpoints/"
    
    ckpt_obj = {
        "tensor1": tensor,
        "model_name": model_name,
        "rng_iterator": 12345,
        "dtype": tensor_dtype,
        "shape": tensor_shape,
        "random_np_obj": np_array
    }
    
    print(f"Engine initalized.. Going to checkpoint now...")

    ckpt_engine.save(state_dict=ckpt_obj, path=ckpt_path)
    print("Async save operation launched")

    ckpt_engine.wait()

    time.sleep(5) # sleep to ensure ckpt file is written
    print(f"Sleep for 5s complete...")      

    recovered_obj = ckpt_engine.load(path=ckpt_path)
    print(f"Checkpoint recovered successfully")
    del ckpt_engine
    
if __name__ == "__main__":
    test_datastates()


