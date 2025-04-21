# weakckpt

Weak consistency checkpoint, which is suitable for classical DNN models, provides a stride saving mechanism and a multi-priority checkpoint recovery mechanism. By improving the efficiency of saving checkpoint parameters and reducing the recovery time caused by training interruption, the overall end-to-end training speed is improved.

Since our research topic has not been published yet, only part of the core code is shown.

...to be continued

### Install and test

```bash
git clone https://github.com/BingxuZhu/weakckpt.git
cd weakckpt/
pip install -r requirement.txt

# Test with a simple DeepSpeed code.
python datastates/tests/test_datastates_llm.py  
```

### Linking with DeepSpeed

please use fork of DeepSpeed at https://github.com/DataStates/DeepSpeed/tree/dev.