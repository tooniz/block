import torch

def block_tensor(tensor, block_dim = 128, ublock_dim = 64, tile_dim = 32, face_dim = 16):
    blocks_r = int(tensor.shape[-2] / block_dim)
    blocks_c = int(tensor.shape[-1] / block_dim)
    ublocks_r = int(block_dim / ublock_dim)
    ublocks_c = int(block_dim / ublock_dim)
    tiles_r = int(ublock_dim / tile_dim)
    tiles_c = int(ublock_dim / tile_dim)
    faces_r = int(tile_dim / face_dim)
    faces_c = int(tile_dim / face_dim)
    blocked_tensor = tensor.reshape(blocks_r,ublocks_r,tiles_r,faces_r,face_dim,blocks_c,ublocks_c,tiles_c,faces_c,face_dim)
    permuted = blocked_tensor.permute(-10,-5,-9,-4,-8,-3,-7,-2,-6,-1)
    flattened = permuted.flatten(start_dim=-10,end_dim=-1)
    back_2d = flattened.reshape(tensor.shape[-2], tensor.shape[-1])
    return back_2d

def unblock_tensor(tensor, block_dim = 128, ublock_dim = 64, tile_dim = 32, face_dim = 16):
    blocks_r = int(tensor.shape[-2] / block_dim)
    blocks_c = int(tensor.shape[-1] / block_dim)
    ublocks_r = int(block_dim / ublock_dim)
    ublocks_c = int(block_dim / ublock_dim)
    tiles_r = int(ublock_dim / tile_dim)
    tiles_c = int(ublock_dim / tile_dim)
    faces_r = int(tile_dim / face_dim)
    faces_c = int(tile_dim / face_dim)
    blocked_tensor = tensor.reshape(blocks_r,blocks_c,ublocks_r,ublocks_c,tiles_r,tiles_c,faces_r,faces_c,face_dim,face_dim)
    permuted = blocked_tensor.permute(-10,-8,-6,-4,-2,-9,-7,-5,-3,-1)
    flattened = permuted.flatten(start_dim=-10,end_dim=-1)
    back_2d = flattened.reshape(tensor.shape[-2], tensor.shape[-1])
    return back_2d
