# 3D_Pose_Estimation
Efficient Pose 방식으로 PyTorch, PyTorch_Lightning framework 사용하여 3D Pose Estimation 진행하는 프로젝트입니다.

## Implementations

- Efficient Pose
- Data (Prepare)
- Data Augmentations (Albumentations)
- Multi GPU Training

## 프로젝트 구조
'''
├─ .gitignore
├─ __README.md

'''

## Requirements
`requirements.txt` 파일을 참고하여 Anaconda 환경 설정 (conda install 명령어)  
`PyYaml`  
`PyTorch`  
`Pytorch Lightning`

## Requirements
`requirements.txt` 파일을 참고하여 Anaconda 환경 설정 (conda install 명령어)  
`PyYaml`  
`PyTorch`  
`Pytorch Lightning`

## Config Train Parameters

기본 설정값은 ./configs/default_settings.yaml에 정의됩니다.  
Train 스크립트 실행 시 입력되는 CFG 파일로 하이퍼파라미터 및 학습 기법을 설정할 수 있습니다.

[default_settings.yaml](./configs/default_settings.yaml)

    // ./configs/*.yaml 파일 수정
    // ex) cls_frostnet -> default_settings 파라미터를 업데이트 해서 사용
    model : 'DarkNet53'
    dataset_name : Poscal_VOC
    classes : 20
    epochs: 500
    data_path : '/mnt/det_train/'
    save_dir : './saved'
    workers: 8
    ...