# AscendNPU-IR Setup on the A5 Server

This note is for the A5 machine workflow: install/runtime setup, Triton Python
lowering, and production of compiler dumps from an A5 environment.

For the non-A5 Codex-server build/replay workflow, see
`NPUIR/coding-guide/codex-server-build.md`.

## 0. SSH to Server

```bash
ssh <user_id>:10.175.112.129
bash

export http_proxy='http://<user_id>:<user_pass>@proxyhk.huawei.com:8080'
export https_proxy=$http_proxy
export no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local
```

## 1. Clone the repository

```bash
cd "$HOME"
git clone https://gitcode.com/wilsoncxfeng/AscendNPU-IR.git
cd AscendNPU-IR
```

Initialize all submodules:

```bash
GIT_SSL_NO_VERIFY=true git submodule update --init --recursive
```

Verify:

```bash
git submodule status --recursive
```

## 2. Create a clean Conda build environment

```bash
conda create --override-channels -c conda-forge \
  -n npuir python=3.10 "cmake>=3.28" "ninja>=1.12" -y

conda activate npuir
```

## 3. Install standalone Clang 15

```bash
cd "$HOME"

mkdir -p "$HOME/tmp"

curl -k -L \
  -o "$HOME/tmp/llvm-15.0.7-x86_64.tar.xz" \
  https://www.kernel.org/pub/tools/llvm/files/llvm-15.0.7-x86_64.tar.xz

mkdir -p "$HOME/tools/llvm-15"

tar -xJf "$HOME/tmp/llvm-15.0.7-x86_64.tar.xz" \
  -C "$HOME/tools/llvm-15" \
  --strip-components=1
```

Verify:

```bash
"$HOME/tools/llvm-15/bin/clang" --version
"$HOME/tools/llvm-15/bin/clang++" --version
"$HOME/tools/llvm-15/bin/ld.lld" --version
```

## 4. Build AscendNPU-IR

```bash
conda activate npuir
cd "$HOME/AscendNPU-IR"

unset CC CXX CFLAGS CXXFLAGS

./build-tools/build.sh \
  -r \
  -o ./build \
  --build-type Release \
  --c-compiler "$HOME/tools/llvm-15/bin/clang" \
  --cxx-compiler "$HOME/tools/llvm-15/bin/clang++" \
  -j 96
```

A successful build ends with:

```text
Build Done!!!
```

## 5. Load the A5 runtime environment

Clone and activate the prepared Python environment:

```bash
mkdir -p "$HOME/.venv"

virtualenv-clone \
  /data/ci_env/daily/python/python3.11_venv \
  "$HOME/.venv/python3.11_venv"

source "$HOME/.venv/python3.11_venv/bin/activate"
```

Load the shared CANN 9.1 environment:

```bash
source /data/pri/Ascend/9.1.0.B087/cann-9.1.0/set_env.sh
```

Verify the A5 device:

```bash
python -c "import acl; print(acl.get_soc_name())"
```

Expected result:

```text
Ascend950PR_9589
```

## 6. Use and test the local compiler

```bash
export PATH="$HOME/AscendNPU-IR/build/install/bin:$PATH"

which bishengir-compile
which bishengir-opt
```

Run a basic test:

```bash
cd "$HOME/AscendNPU-IR/build"

./bin/llvm-lit \
  ../bishengir/test/bishengir-compile/commandline.mlir \
  -v
```

Expected result:

```text
Passed: 1 (100.00%)
```

## 7. Add Environment Helpers to `~/.bashrc`

```bash
export ASCENDNPU_IR_ROOT="$HOME/AscendNPU-IR"
export LLVM15_HOME="$HOME/tools/llvm-15"
export PATH="$ASCENDNPU_IR_ROOT/build/install/bin:$LLVM15_HOME/bin:$PATH"

export TRITON_ASCEND_ARCH=Ascend910_9589
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export TRITON_DUMP_DIR="$HOME/.triton/dump"
export TRITON_ALWAYS_COMPILE=1 #might need to remove this for non-debug mode 
export TRITON_DEBUG=1 #might need to remove this for non-debug mode

npuir-build-env() {
    conda activate npuir
    export PATH="$LLVM15_HOME/bin:$PATH"
    export CC="$LLVM15_HOME/bin/clang"
    export CXX="$LLVM15_HOME/bin/clang++"
}

npuir-runtime-env() {
    source "$HOME/.venv/python3.11_venv/bin/activate"
    source /data/pri/Ascend/9.1.0.B087/cann-9.1.0/set_env.sh
    export PATH="$ASCENDNPU_IR_ROOT/build/install/bin:$PATH"
}
```

Reload:

```bash
source "$HOME/.bashrc"
```

## 8. Run Triton and Dump AscendNPU-IR Passes

Enable Triton recompilation and dumps:

```bash
npuir-runtime-env

mkdir -p "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR"

python my-example.py
```

Locate the generated IR:

```bash
find "$TRITON_DUMP_DIR" -type f -name '*.mlir' | sort
```

Go to that directory. Copy the `[DEBUG] cmd_list` printed by Triton, modify the addresses to point to the current dump. Remove `--mlir-print-ir-after-failure --bishengir-print-ir-after=hivm-graph-sync-solver` then add:

```bash
--mlir-disable-threading  --mlir-print-ir-after-all
```

Something like:

```bash
bishengir-compile ./kernel.ttadapter.mlir --target=Ascend950PR_9589 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-hfusion-compile=true --enable-triton-kernel-compile=true  -o "$HOME/tmp/npuir-replay/kernel" --enable-vf-merge-level=1 --mlir-disable-threading  -mlir-print-ir-after-all &> after_all.mlir
```

Redirect the output:

```bash
> all_passes.log 2>&1
```

List pass boundaries:

```bash
grep -n "IR Dump" all_passes.log
```
