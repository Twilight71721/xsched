#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st* CUstream;

XResult CudaQueueCreate(HwQueueHandle *hwq, CUstream stream);

XResult CudaQueueGet(HwQueueHandle *hwq, CUstream stream);

// Process-local owner of a CHANNEL LV3 compute group and its scheduler queue.
typedef struct CudaTaskDomain CudaTaskDomain;
typedef XResult (*CudaTaskWork)(CUstream stream, void *argument);

XResult CudaTaskDomainCreate(CudaTaskDomain **domain, int device_ordinal);
// Caller dedicates the whole normal context to this task and retains the idle stream.
XResult CudaTaskDomainAttach(CudaTaskDomain **domain, CUstream stream);
XResult CudaTaskDomainGetQueue(const CudaTaskDomain *domain, XQueueHandle *xq);
// Invoke host work with the task context current. Submit CUDA work on the supplied stream.
XResult CudaTaskDomainWithStream(CudaTaskDomain *domain, CudaTaskWork work, void *argument);
// Stop new submissions and join host users before destruction. Clears the handle on success.
XResult CudaTaskDomainDestroy(CudaTaskDomain **domain);

#ifdef __cplusplus
}
#endif
