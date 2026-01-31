# include目录如下图所示：

<img src="1.assets/image-20260131100137881.png" alt="image-20260131100137881" style="zoom:60%;" />



# 在LLAISYS里，“层级”可以这样理解：

```mathematica
┌────────────────────────────┐
│ Python / Model / generate  │  ← 变化快、语义高
├────────────────────────────┤
│   Ops API（llaisys_ops.h） │
├────────────────────────────┤
│ Runtime C ABI（llaisys.h） │  ← 最底层【对外】
├────────────────────────────┤
│ C++ Tensor / Device / Kern │
├────────────────────────────┤
│ CUDA / CPU / Driver        │
└────────────────────────────┘
```

## llaisys.h

`llaisys.h` 是 **LLAISYS Runtime 的最底层 C ABI 接口定义文件**。

1. 这里的**最底层**指的是：所有对外API的地基，其他所有对外接口（`ops/tensor`）都依赖他，但它不依赖任何上层语义。一旦`llaisys.h`变化，位于它上层的所有东西都要重编、重绑定、重适配。

2. **ABI**（Application Binary Interface）：“编译后，二进制层面如何互相调用”。它规定了函数名长什么样、参数如何压栈/传寄存器、struct/enum在内存里的布局，谁负责释放内存。

3. 为什么是**C ABI**：因为C 是“事实上的跨语言最小公分母”，几乎所有的语言都能调用C ABI（例如`python、rust、java、C++`等）。而且C++ ABI 是不稳定的，体现在不同编译器不一样、不同版本不一样、name mangling不可控。所以对外接口必须使用C ABI。

4. **Runtime**在LLASYS中的定义是：负责把计算正确地、在正确设备上执行出来。

|      责任       |        举例        |
| :-------------: | :----------------: |
| Tensor 生命周期 | 分配 / 释放 / 引用 |
|   Device 管理   |     CPU / GPU      |
|   Kernel 调度   |   用哪个 kernel    |
|    内存拷贝     |     H2D / D2H      |
|    执行语义     |    同步 / 异步     |

​    Runtime不关心模型语义，只关心“如何跑”。

5. 为什么它是**接口定义文件**？

   因为它只做三件事：

   1️⃣  声明类型（enum / handle）

   2️⃣  生命调用规则（extern "C"）

   3️⃣  声明可见性（__export）

   它不包含任何实现，“我承诺你可以这样调用我，但我没告诉你我是怎么实现的。”

6. 为什么需要接口定义文件？

   如果没有接口定义文件，会发生什么？

   ❌1️⃣  ：直接暴露 C++ 类

   ```C++
   class Tensor {
     std::vector<int64_t> shape;
     ...
   };
   ```

   这样的话：Python调用不了，ABI不稳定，改一个成员，全部崩

   ❌ 2️⃣：头文件里全是实现

   这样的话：编译慢、耦合高、无法隐藏内部设计、无法替换实现

7. 为什么接口定义文件要放在这个头文件里？

   `llaisys.h` 的内容特点是：✅ 不依赖 C++、✅ 不依赖 Tensor 实现、✅ 不依赖算子语义、✅ 不依赖 STL、✅ 不依赖第三方库，它是 **“零负担可包含头文件”**

8. 代码

   ```C
   #ifndef __LLAISYS_H__//头文件保护
   #define __LLAISYS_H__
   
   //__export：跨平台“符号导出”宏
   #if defined(_WIN32)
   #define __export __declspec(dllexport)
   #elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
   #define __export __attribute__((visibility("default")))
   #else
   #define __export
   
   #endif//头文件保护，防止头文件被重复包含导致重复定义错误，这是所有公共头文件的标配
   #ifdef __cplusplus
   #define __C extern "C"//为什么需要 extern "C"？C++ 有 name mangling（名字改编），导致在C++编译后C或者python不认识编译后的新名字，所以extern "C" void foo(int);就会强制使用 C ABI。
   #include <cstddef>
   #include <cstdint>
   #else
   //__C：C / C++ ABI 兼容，让这个头文件既能被 C 编译器用，也能被 C++ 编译器用，__C 的真实含义是：“以下声明是 C ABI，不要做 C++ 名字改编”
   #define __C
   #include <stddef.h>
   #include <stdint.h>
   #endif
   
   // Device Types：设备抽象，这是一个运行时设备抽象层。
   typedef enum {
       LLAISYS_DEVICE_CPU = 0,
       LLAISYS_DEVICE_NVIDIA = 1,
       LLAISYS_DEVICE_TYPE_COUNT
   } llaisysDeviceType_t;
   
   // Data Types：统一的数据类型系统，Runtime 层的 dtype 系统
   typedef enum {
       LLAISYS_DTYPE_INVALID = 0,
       LLAISYS_DTYPE_BYTE = 1,
       LLAISYS_DTYPE_BOOL = 2,
       LLAISYS_DTYPE_I8 = 3,
       LLAISYS_DTYPE_I16 = 4,
       LLAISYS_DTYPE_I32 = 5,
       LLAISYS_DTYPE_I64 = 6,
       LLAISYS_DTYPE_U8 = 7,
       LLAISYS_DTYPE_U16 = 8,
       LLAISYS_DTYPE_U32 = 9,
       LLAISYS_DTYPE_U64 = 10,
       LLAISYS_DTYPE_F8 = 11,
       LLAISYS_DTYPE_F16 = 12,
       LLAISYS_DTYPE_F32 = 13,
       LLAISYS_DTYPE_F64 = 14,
       LLAISYS_DTYPE_C16 = 15,
       LLAISYS_DTYPE_C32 = 16,
       LLAISYS_DTYPE_C64 = 17,
       LLAISYS_DTYPE_C128 = 18,
       LLAISYS_DTYPE_BF16 = 19,
   } llaisysDataType_t;
   
   // Runtime Types：“不透明的运行时句柄”，外部用户不能也不应该知道内部结构，对外暴露为 void*
   typedef void *llaisysStream_t;
   
   // Memory Copy Directions：内存拷贝方向
   typedef enum {
       LLAISYS_MEMCPY_H2H = 0,
       LLAISYS_MEMCPY_H2D = 1,
       LLAISYS_MEMCPY_D2H = 2,
       LLAISYS_MEMCPY_D2D = 3,
   } llaisysMemcpyKind_t;
   
   #endif // __LLAISYS_H__
   ```

## ops.h

定义了 LLAISYS Runtime 对外暴露的“**算子级 C ABI 接口**”

1. 代码

   ```c
   #ifndef LLAISYS_OPS_H
   #define LLAISYS_OPS_H
   
   #include "tensor.h"//，引入唯一合法的计算对象：llaisysTensor_t，在LLAISYS中，一切算子的输入/输出必须是Tensor句柄。这是Runtime对控制权的声明：不允许裸传内存指针、不允许外部控制shape/stride/device，所有合法性检查必须在Runtime内完成。
   __C {//使用_C指明：以下声明是 C ABI，不要做 C++ 名字改编
       __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);//本质是extern "C" __declspec(dllexport) void llaisysAdd(...);
       __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
       __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
       __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
       __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
       __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
       __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
       __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
       __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
   }
   
   #endif
   ```

2. 为什么要把这些函数**放在这里**？

   1️⃣ 为什么不放在 `llaisys.h` 里？

   `llaisys.h` 的职责是：定义“运行时的基础世界观”（设备是什么、dtype有哪些、stream是什么、memcpy语义是什么），它不应该知道任何模型语义，而`ops.h`这里的`Add / Attention / Linear` 明显是 **模型层概念**。所以必须拆出来。

   2️⃣ 为什么不放在 `tensor.h` 里？

   `tensor.h`通常只负责：Tensor 是什么、Tensor 怎么创建 / 销毁、Tensor 的元信息接口。Tensor 是“数据容器”，不是“计算语义”。把算子放进去会造成 **职责污染**。

   3️⃣ 为什么不用 C++ 接口（class / method）？

   如果写成`Tensor::add(const Tensor&);`那么，Python ctypes 不能用、ABI 不稳定、后续 device / kernel 演进成本极高，所以 Runtime 层 **必须用 C ABI**。

3. 它在整个项目中的地位

   1️⃣ 架构定位图

   ```mathematica
   ┌──────────────────────────────┐
   │ Python / generate / Model    │
   │ (模型逻辑，for 循环)           │
   ├──────────────────────────────┤
   │ llaisys_ops.h                │  ←【你现在问的】
   │ 算子级 C ABI（模型语义）        │
   ├──────────────────────────────┤
   │ llaisys.h                    │
   │ Runtime 基础 C ABI            │
   ├──────────────────────────────┤
   │ Tensor / Device / Stream     │
   │ (C++ 实现层)                  │
   ├──────────────────────────────┤
   │ Kernel / CUDA / CPU          │
   └──────────────────────────────┘
   
   ```

   2️⃣ 它是“模型与 Runtime 的分界线”

   可以理解成：“模型世界对 Runtime 的唯一合法入口”。因为模型代码不能直接操作Tensor内存、直接调用CUDA kernel、直接访问device细节。它只能：

   ```c
   llaisysLinear(...)
   llaisysSelfAttention(...)
   ```

   3️⃣ 它是“稳定 API 层”，不是实验区

   如果这个文件变化，则python绑定要改、文档要改、所有模型代码要改。所以它的设计一定是少而稳。所以它是接口定义文件，而不是实现文件。这正好符合Runtime的设计原则：“接口稳定，内部实现可以随意推翻重写”

## runtime.h

这是**“Runtime 能力本身的抽象入口”**，它定义了 LLAISYS Runtime 的“函数表（function table）接口”，用于在运行时按设备类型（CPU / NVIDIA）获取一整套底层执行能力。

1. 代码

   ```C
   #ifndef LLAISYS_RUNTIME_H
   #define LLAISYS_RUNTIME_H
   
   #include "../llaisys.h"
   
   __C {
       // Runtime API Functions，1️⃣ 它引入了 Runtime 级别的“能力接口”，而不是函数实现，不是__export void llaisysMallocDevice(...);而是typedef void *(*malloc_device_api)(size_t);也就是说：Runtime 的能力不是通过“全局函数”暴露的，而是通过“函数指针表”暴露的。
       
   //2️⃣ 它把 Runtime 能力分成了4类，这四类正好构成一个最小可用Runtime。
       // 🔹Device管理：控制“当前执行设备上下文”
       typedef int (*get_device_count_api)();
       typedef void (*set_device_api)(int);
       typedef void (*device_synchronize_api)();
       // 🔹Stream管理：控制“异步执行语义”
       typedef llaisysStream_t (*create_stream_api)();
       typedef void (*destroy_stream_api)(llaisysStream_t);
       typedef void (*stream_synchronize_api)(llaisysStream_t);
       // 🔹Memory内存管理：Runtime 自己掌控内存，而不是交给外部
       typedef void *(*malloc_device_api)(size_t);
       typedef void (*free_device_api)(void *);
       typedef void *(*malloc_host_api)(size_t);
       typedef void (*free_host_api)(void *);
       // 🔹Memory copy内存拷贝：统一 CPU / GPU / Stream 的拷贝语义
       typedef void (*memcpy_sync_api)(void *, const void *, size_t, llaisysMemcpyKind_t);
       typedef void (*memcpy_async_api)(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t);
       
   //3️⃣ struct LlaisysRuntimeAPI = Runtime 的“能力表”，这在架构上等价于：CUDA 的 cudaDriverGetProcAddress、Vulkan 的 VkDeviceDispatchTable、ONNX Runtime 的 OrtApi。它的语义是：“你拿到这张表，就等于拿到了 Runtime 的全部底层能力。”
       struct LlaisysRuntimeAPI {
           get_device_count_api get_device_count;
           set_device_api set_device;
           device_synchronize_api device_synchronize;
           create_stream_api create_stream;
           destroy_stream_api destroy_stream;
           stream_synchronize_api stream_synchronize;
           malloc_device_api malloc_device;
           free_device_api free_device;
           malloc_host_api malloc_host;
           free_host_api free_host;
           memcpy_sync_api memcpy_sync;
           memcpy_async_api memcpy_async;
       };
   
       //4️⃣ Llaisys API for getting the runtime APIs，这是关键入口函数。它的语义是：“给我一个设备类型，我返回一套对应设备的 Runtime 能力实现。” 例如：LLAISYS_DEVICE_CPU → CPU Runtime、LLAISYS_DEVICE_NVIDIA → CUDA Runtime
       __export const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t);
   
       //5️⃣ Llaisys API for switching device context，这是上下文切换接口。它的作用是：选择当前 Runtime、选择当前 device id、影响后续所有 Tensor / Kernel / memcpy 行为。这是 Runtime 的全局上下文入口。
       __export void llaisysSetContextRuntime(llaisysDeviceType_t, int);
   }
   
   #endif // LLAISYS_RUNTIME_H
   
   ```

   

