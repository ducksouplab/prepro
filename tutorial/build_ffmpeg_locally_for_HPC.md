# Local Installation of FFmpeg + x264 on an HPC (No Root)

Below is one **complete** Markdown file. You can copy/paste it directly into a repository **as is**. It explains how to install [NASM](https://www.nasm.us/), [x264](https://www.videolan.org/developers/x264.html), and [FFmpeg](https://ffmpeg.org/) into a **user-only directory** (`/mnt/data/project0028`) on an HPC system (no root privileges).  
This yields a **fully functional** FFmpeg with H.264 encoding via `libx264`.

---

## Table of Contents
1. [Overview](#overview)  
2. [Environment Setup](#environment-setup)  
3. [Build & Install NASM](#build--install-nasm)  
4. [Build & Install x264](#build--install-x264)  
5. [Build & Install FFmpeg](#build--install-ffmpeg)  
6. [Check Encoders](#check-encoders)  
7. [Usage](#usage)  
8. [Troubleshooting Tips](#troubleshooting-tips)

---

## Overview

We will:

1. **Set environment variables** for our local install prefix (`/mnt/data/project0028`).  
2. **Build & install NASM** (the assembler required by x264).  
3. **Build & install x264** (for H.264 software encoding).  
4. **Build & install FFmpeg** (with libx264).  

After these steps, you’ll have a **self-contained** FFmpeg in `/mnt/data/project0028/bin/ffmpeg`, with full H.264 support.

---

## Environment Setup

1. **Define** your install directory.  
2. **Create** a folder for source code.  
3. **Add** the local bin directory to your PATH.  
4. **Export** PKG_CONFIG_PATH so pkg-config can detect our local libraries.

~~~bash
# 1) Define your installation prefix
export INSTALL_DIR="/mnt/data/project0028"

# 2) Create a place to store source code
export SRC_DIR="$INSTALL_DIR/ffmpeg_sources"
mkdir -p "$SRC_DIR"

# 3) Update PATH to include local bin
export PATH="$INSTALL_DIR/bin:$PATH"

# 4) Let pkg-config see our local .pc files
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig"
~~~

*(You can add these exports to your `~/.bashrc` if you want them always applied.)*

---

## Build & Install NASM

x264 requires **NASM ≥ 2.13** for assembly optimizations. Compile NASM locally:

~~~bash
cd "$SRC_DIR"

# Example: download NASM 2.15.05
wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz

tar -xvf nasm-2.15.05.tar.gz
cd nasm-2.15.05

./configure --prefix="$INSTALL_DIR"
make -j"$(nproc)"
make install

# Check that nasm is now in PATH & is correct version
which nasm
nasm -v
~~~

---

## Build & Install x264

Now compile **x264**, which uses NASM for its fast assembly routines:

~~~bash
cd "$SRC_DIR"

# Clone the stable branch
git clone --branch stable https://code.videolan.org/videolan/x264.git
cd x264

./configure \
  --prefix="$INSTALL_DIR" \
  --enable-static \
  --disable-opencl

make -j"$(nproc)"
make install

# Verify x264
$INSTALL_DIR/bin/x264 --version
~~~

---

## Build & Install FFmpeg

Finally, build FFmpeg **with** `libx264` support:

~~~bash
cd "$SRC_DIR"
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

./configure \
  --prefix="$INSTALL_DIR" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$INSTALL_DIR/include" \
  --extra-ldflags="-L$INSTALL_DIR/lib" \
  --extra-libs="-lpthread -lm" \
  --enable-gpl \
  --enable-libx264 \
  --disable-shared \
  --enable-static \
  --disable-debug

make -j"$(nproc)"
make install

# Verify ffmpeg
$INSTALL_DIR/bin/ffmpeg -version
~~~

You should see `--enable-libx264` in the configuration line.

---

## Check Encoders

Make sure FFmpeg recognizes the x264 encoder:

~~~bash
$INSTALL_DIR/bin/ffmpeg -encoders | grep 264
~~~

Expected output (varies slightly by version): V....D libx264 libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264) V....D libx264rgb libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 RGB (codec h264) ...


---

## Usage

You can now run your locally installed FFmpeg explicitly:

~~~bash
/mnt/data/project0028/bin/ffmpeg -i input.mp4 -c:v libx264 output.mp4
~~~

Or, if you’ve updated your `PATH`:

~~~bash
export PATH="/mnt/data/project0028/bin:$PATH"
ffmpeg -i input.mp4 -c:v libx264 output.mp4
~~~

---

## Troubleshooting Tips

1. **“Found no assembler” Error**  
   - Ensure NASM ≥ 2.13 is correctly installed and in `$PATH`.  

2. **wget/git Not Available**  
   - Manually upload tarballs or `.git` repos to the HPC, then build locally.  

3. **Environment Variables**  
   - Re-export your `PATH` and `PKG_CONFIG_PATH` if you start a new shell session.  

4. **Performance**  
   - Confirm `nasm` is used so x264 is compiled with assembly optimizations for speed.

---

**Done!**  
You have a **local** install of NASM, x264, and FFmpeg (with libx264) in `/mnt/data/project0028`, requiring **no root privileges** on your HPC.




