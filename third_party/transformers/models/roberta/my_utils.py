import numpy as np
import torch
import copy

#iput_emb: [bsz,max_len,D]
#input mask: [bsz,max_len] Ex: [1, 2, 3,3,4,5]
def convert_subtok_to_tok(input_emb, input_mask):
	max_num_pos_words = torch.amax(input_mask,dim=-1)
	max_num_pos_words += 1
	device = input_emb.get_device()
	final_max = torch.amax(max_num_pos_words,0).item()
	max_seq_len = input_emb.shape[1]
	out_tensor = []
	out_no_extra=[]
	#print ("max num", max_num_pos_words)

	#print ("Mask check", input_mask[6])
	for idx in range (input_emb.shape[0]): #go through each sample
		max_num = max_num_pos_words[idx].item()	
		cur_emb = input_emb[idx] # max_len,D
		cur_mask = input_mask[idx] #[max_len]
		replacement=[]
		replacement.append(cur_emb[0:1]) #[CLS] position
		replacement_no_extra=[]
		for it in range (max_num):
			loc = np.nonzero(cur_mask == it)
			emb_int = cur_emb[cur_mask == it] #[num_subtoken,D]
			# Issue is when one of the 768 dimensions is 0, discrepancy exists ==> more complicated version  is not needed!
			# Q: is it okay for [num_subtoken] vectors containing all 0s? In that case, it is a mask
			#old_mean = emb_int.mean(dim=0,keepdim=True)
			#new_mean = emb_int.sum(dim=0,keepdim=True) / (torch.count_nonzero(emb_int, dim=0) + 1e-20)
			#try:	
			#assert torch.allclose(old_mean,new_mean)
			#assert torch.count_nonzero(emb_int.sum(dim=1,keepdim=True),dim=0).all() 
			#except:
			#	print ("prior emb-int", emb_int.shape)
			#	print ("What to check", torch.count_nonzero(emb_int.sum(dim=1,keepdim=True),dim=0))
			#	print ("Check", torch.count_nonzero(emb_int,dim=0), emb_int.sum(dim=0, keepdim=True))
			#	print ("Loc", idx, it, cur_mask, loc)

			#print ("Example", torch.count_nonzero(emb_int.sum(dim=-1),dim=0))	
			#assert torch.count_nonzero(emb_int.sum(dim=-1), dim=0).all() != 0
			#emb_int = emb_int.mean(dim=0,keepdim=True)
			emb_int = emb_int.sum(dim=0,keepdim=True) / (torch.count_nonzero(emb_int.sum(dim=-1), dim=0)+1e-20)

			
			#assert torch.allclose(old_mean,new_mean)
			#emb_int = emb_int[0:1]
			replacement.append(emb_int)
			replacement_no_extra.append(emb_int)
		
		#Add SEP loc 
		replacement.append(cur_emb[loc[-1].item()+1:loc[-1].item()+2])
		replacement = torch.cat(replacement,0)

		replacement_no_extra = torch.cat(replacement_no_extra,0)

		zero_tensor = torch.zeros(max_seq_len - replacement.shape[0], cur_emb.shape[-1]).to(device)
		replacement = torch.cat([replacement, zero_tensor], 0)
	
		zero_tensor_extra = torch.zeros(max_seq_len - replacement_no_extra.shape[0], cur_emb.shape[-1]).to(device)
		replacement_no_extra = torch.cat([replacement_no_extra, zero_tensor_extra], 0)
		assert replacement.shape[0] == max_seq_len

		assert replacement_no_extra.shape[0] == max_seq_len
		out_tensor.append(replacement)
		out_no_extra.append(replacement_no_extra)

	out_tensor = torch.stack(out_tensor,dim=0) #[bsz,max_len,D]
	out_no_extra = torch.stack(out_no_extra, dim=0)

	# Sanity Check
	tensor_shrink = out_tensor.sum(dim=-1) #[bsz,max_len]
	count_torch = torch.count_nonzero(tensor_shrink, dim=-1) #[num_tokens + sep_token]
	count_torch -= 2 #-separate_tokens (-2 for BERT, -3 XLM-R)
	#print ("Count torch (from new tensor)", count_torch)
	#print ("Max num possible words (from max mask)", max_num_pos_words)
	#assert torch.allclose(count_torch, max_num_pos_words)

	##Sanity check (for no extra CLS and SEP)
	#tensor_no_extra= out_no_extra.sum(dim=-1)
	#count_no_extra  = torch.count_nonzero(tensor_no_extra,dim=-1)
	#assert torch.allclose(count_no_extra, max_num_pos_words)

	#test_mask = input_mask[8]

	#print ("Input mask", input_mask[8])
	#test_mask[test_mask !=-1] = 1
	#print ("First after", test_mask)
	#test_mask[test_mask ==-1] = 0

	#print ("second after", test_mask)
	#print ("Test mask", test_mask.sum())


	#except:
	#	print (count_torch, max_num_pos_words)
	#	print ("Tensor shrink", tensor_shrink)
	#	print ("Input mask", input_mask.sum(dim=-1))
	#	int_idx=2
	#	print ("Shrink tensor", tensor_shrink[int_idx])
	#	print ("Input mask", input_mask[int_idx])
	#	test_mask = input_mask[int_idx]
	#	test_mask[test_mask !=-1] = 1
	#	test_mask[test_mask ==-1] = 0
	#	print ("Test mask", test_mask.sum())

	return out_tensor, out_no_extra
			
