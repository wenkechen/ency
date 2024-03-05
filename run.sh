#!/usr/bin/env bash

function delete_file() {
	file=$1
	if [ -e $file ]; then
		rm $file
	fi
}

function delete_cmake_files() {
	dir=$1
	rm -rf $dir/CMakeFiles
	$(delete_file $dir/cmake_device_link.compute_80.cubin)
	$(delete_file $dir/cmake_device_link.compute_86.cubin)
	$(delete_file $dir/cmake_device_link.fatbin)
	$(delete_file $dir/cmake_device_link.fatbin.c)
	$(delete_file $dir/cmake_device_link.reg.c)
	$(delete_file $dir/CMakeCache.txt)
	$(delete_file $dir/cmake_install.cmake)
	$(delete_file $dir/Makefile)
	$(delete_file $dir/detect_cuda_version.cc)
}

if [ $1 = "lint" ]; then
	bash scripts/lint.sh
elif [ $1 = "clear" ]; then
	$(delete_cmake_files "./csrc")
	$(delete_cmake_files "./csrc/tests")
elif [ $1 = "build" ]; then
	root_path=`pwd`
	echo "Building csrc..."
	cd $root_path/csrc
	cmake . -DBUILD_TH=ON
	cmake --build . -- -j$(nproc)
	cd $root_path
fi
