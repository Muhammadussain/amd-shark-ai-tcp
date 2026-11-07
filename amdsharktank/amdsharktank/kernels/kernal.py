from .base import *

import torch 
__all__ =[
    "mmt_block_scaled_q5_0",
]

@CustomOp.register(library=LIBRARY)
class mmt_block_scaled_q5_0(CustomOp):
    
    signature = (
        "mmt_block_scaled_q5_0(Tensor a,Tensor d,Tensor qh, Tensor qs)  ->(Tensor)"
        )
    
    def select(self,ksel :KernelSelection):
       a_desc  = ksel.arg_tensor(0)
       d_desc  = ksel.arg_tensor(1)
       qh_desc = ksel.arg_tesor(2)
       qs_desc = ksel.ag_tensor(3)
       *batch_dims ,a_m ,a_k =a_desc.t.shape
    torch._check(
           a_desc.t.dtype.is_floating_point,
           lambda:f"mmt_block_scaled_q5_0 arg 'a' :Expected floating poit (got{a_desc.t.dtype})",
       )
    torch._check(
            len(batch_dims) ==1,
            lambda :f"mmt_block_scaled_q5_0 arg 'a' Expected 3d tensor got {a_desc.t.shape}",
             
        )
        
        qh_n ,qh_group0 ,qh_bs  ,*rest = qh_desc.t.shape
    torch.check(
            len(rest)==0 and qh_bs == 4 and qh_group0 == qs_group and qh_n ==qs_n,
            lambda :f"mmt_blocked_scaled_q5_0 arg 'qh' Incorrrect shape (got {qh_desc.t.shape})",  
        )
        d_n , d_group0 ,d_one ,*rest =d_desc.t.shape 
        torch._check (
            len(rest) ==0 and d_group ==qs_group and d_one == 1 and d_n qs_n,
            lambda :f"mmt_block_scaled_q5_0 arg 'd' :Incorrect shape got {d_desc.t.shape}"
        )
        a_desc.specialize_dims(-1)
        d_desc.specialize_dims()
        qh_bs.specialize_dims()
        qs_bs.specialize_dims()
        
        
        c_desc =ksel.return_new_tensor(batch_dims +[a_m,d_n] ,d_type =a_desc.t.type)
        c_desc.specialize_dims(-1)
        
        
        