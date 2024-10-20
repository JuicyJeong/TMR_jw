
from pytorch_lightning import seed_everything
import hydra
import torch
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
from encode_motion_jw import encode_motion_jw
from concurrent.futures import ThreadPoolExecutor, as_completed

from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import src.prepare  # noqa
import torch
import numpy as np
from src.config import read_config
from src.load import load_model_from_cfg
from hydra.utils import instantiate
import time
######################################################
import warnings
warnings.filterwarnings("ignore", message="No audio backend is available.")

'''
encoded된 두 파일의 코사인 유사도를 비교하여 반환하는 스크립트.
1. 소스 파일을 로드합니다.
2. 타겟 파일(1개)를 로드합니다.
3. 로드된 소스와 타겟의 파일을 각각 인코더에 집어넣고 출력된 latent vec을 각각 변수에 입력
4. 두 변수의 코사인 유사도를 출력합니다.
    4-1. 유사도가 0.NN이상이면 해당 파일 이름과 프레임 구간을 반환
5. 4-1의 리스트를 저장.

 motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
1. 타겟의 소스를 하나 로드함
2. 프레임 수를 반환.
3. 전체 프레임수랑 내가 나눠서 입력할 프레임 수를 비교함
    3-1. 비교후, 0프레임부터 순차적으로 자를 프레임을 분할
예를 들어, 타겟이 30프레임이고, 나는 20프레임씩 자른다고 할때, 나올 수 있는 가짓수는
0-19, 1-20, 2-21, ... 9-30 이렇게 10가지의 가짓수가 나와. 이걸 리스트에 저장하는 함수를 만들고 싶어.
'''


def get_sliced_motion_segments(npy_path, segment_length):
    motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    total_frames = motion.shape[0]
    print(f'total_frames:{total_frames}')

    segments = []
    if total_frames <= segment_length:
        add_seg = motion
        segments.append(add_seg)
    else:
        for start in range(total_frames - segment_length + 1):
            end = start + segment_length
            add_seg = motion[start:end]
            segments.append(add_seg)

    return segments


# 여러 개의 모델을 병렬적으로 호출하여 배치를 처리하는 함수


def process_segments_parallel(segments, max_workers=4):
    latent_vectors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 병렬로 실행할 작업 제출
        futures = [executor.submit(
            encode_motion, segment) for segment in segments]

        # 작업이 완료되면 결과를 모음
        for future in as_completed(futures):
            latent_vectors.append(future.result())

    return latent_vectors


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

###################################################################################################################
# 인코더 함수 정의


def encode_motion(motion, model, normalizer, device='cuda'):
    from src.data.collate import collate_x_dict

    motion = normalizer(motion)
    motion = motion.to(device)

    motion_x_dict = {"x": motion, "length": len(motion)}

    with torch.inference_mode():
        motion_x_dict = collate_x_dict([motion_x_dict])
        latent = model.encode(motion_x_dict, sample_mean=True)[0]
        latent = latent.cpu().numpy()

    return latent


class MotionDataset(Dataset):
    def __init__(self, data_folder_path, src_frame_len, model, normalizer, device='cuda', skip_frames=1):
        self.data_folder_path = data_folder_path
        self.src_frame_len = src_frame_len
        self.device = device
        self.model = model
        self.normalizer = normalizer
        self.skip_frames = skip_frames  # 스킵할 프레임 수 추가
        self.data_files = self.load_data_files()
        self.index_mapping = self.create_index_mapping()

    def load_data_files(self):
        files = []
        for file_name in os.listdir(self.data_folder_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(self.data_folder_path, file_name)
                files.append(file_path)
        return files

    def create_index_mapping(self):
        index_mapping = []
        current_index = 0
        for file_path in self.data_files:
            data = np.load(file_path)
            file_name = os.path.basename(file_path)
            if len(data) < self.src_frame_len:
                index_mapping.append((file_name, 0, len(data), current_index))
                current_index += 1
            else:
                for start_idx in range(0, len(data) - self.src_frame_len + 1, self.skip_frames):
                    end_idx = start_idx + self.src_frame_len
                    index_mapping.append(
                        (file_name, start_idx, end_idx, current_index))
                    current_index += 1
        return index_mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        file_name, start_idx, end_idx, _ = self.index_mapping[idx]
        file_path = os.path.join(self.data_folder_path, file_name)
        data = np.load(file_path)
        input_tensor = torch.from_numpy(
            data[start_idx:end_idx]).to(torch.float)

        # 인코더를 통해 latent vector 생성
        latent_vector = encode_motion(
            input_tensor, self.model, self.normalizer, device=self.device)
        meta_info = f"{file_name}_{start_idx}_{end_idx}"
        return latent_vector, meta_info


if __name__ == "__main__":

    # 유사도 일정 값 넘어갔을 때의 로그를 csv로 작성
    # 텍스트 파일 초기화
    log_file_name = "log10.csv"
    txt_1st_line = "ref_file_name,src_file_name,Cos_Similarity,max_frame_start,max_frame_end\n"
    txt_nth_line = ""
    with open(log_file_name, "w", encoding="utf-8") as file:
        file.write(txt_1st_line)

    run_dir = 'models/tmr_humanml3d_guoh3dfeats'
    ckpt_name = 'last'
    device = 'cuda:0'

    data_folder_path = "npy/jw_new_joint_vecs"

    # 모델 로드
    cfg = read_config(run_dir)
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    src_npy_path = 'npy/jw_fly_vecs'
    src_files = os.listdir(src_npy_path)

    for i in range(0, 10):  # 70개니깐 알아서 조정해
        file_name = os.path.join(src_npy_path, src_files[i])
        # print(file_name)
        ref_motion = torch.from_numpy(np.load(file_name)).to(torch.float)
        # print(motion.shape)
        src_lengh = ref_motion.shape[0]
        ref_file_name = src_files[i]
        print(f'비행 동작"{ref_file_name}"이 로드되었습니다.')
        print(f'비행동작 길이: {ref_motion.shape[0]}')
        latent_src = encode_motion(ref_motion, model, normalizer)

        # ref_path = 'npy/jw_fly_vecs/'
        # file_name = 'Ascend_Start.npy'
        # ref_motion = torch.from_numpy(np.load(ref_path+file_name)).to(torch.float)
        # ref_latent = encode_motion(ref_motion, model, normalizer)
        # print(f'비행동작 길이: {ref_motion.shape[0]}')
        # print(f'비행동작 latent vec:{ref_latent.shape}')

        time.sleep(1)

        # 데이터셋 생성
        print("데이터셋 로드")
        dataset = MotionDataset(data_folder_path=data_folder_path,
                                src_frame_len=int(ref_motion.shape[0]), model=model, normalizer=normalizer, device=device, skip_frames=4)

        # DataLoader 생성
        print("데이터로더 로드")
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False)

        # 실행 시간 측정
        # start_time = time.time()
        print("배치 시작")
        for batch_latent_vector, batch_meta_info in data_loader:
            print(
                f"Meta Info[0]: {batch_meta_info[0]}, Latent Vector: {batch_latent_vector.shape}")
            for i in range(len(batch_latent_vector)):  # 배치 사이즈 만큼 반복
                sim_score = cos_sim(batch_latent_vector[i], latent_src)
                if sim_score > 0.7:
                    print(f'{batch_meta_info[i]}: {sim_score}')
                    batch_data_list = batch_meta_info[i].split("_")
                    # "ref_file_name,h3d_file_name,Cos_Similarity,max_frame_start,max_frame_end"

                    # input_data = [str(src_files[i]), batch_data_list[0], str(
                    #     sim_score), batch_data_list[1], batch_data_list[2]]
                    # content = ",".join(input_data)
                    content = f'{ref_file_name},{batch_data_list[0]},{sim_score},{batch_data_list[1]},{batch_data_list[2]}\n'
                    txt_nth_line += content
                # print(txt_nth_line)
                with open(log_file_name, "w", encoding="utf-8") as file:
                    file.write(txt_1st_line + txt_nth_line)

        # end_time = time.time()
        # execution_time = end_time - start_time

        # print(f"Total execution time: {execution_time:.2f} seconds")
