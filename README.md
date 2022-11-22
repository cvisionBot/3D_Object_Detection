# 3D Object Detection
PyTorch, PyTorch_Lightning framework 사용하여 3D Object Detection을 진행하는 프로젝트입니다.

## Implementations
- Data (Prepare)
- Multi GPU Training

## 프로젝트 구조
'''
├─ dataset
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

## TODOs
- Data Generator
- Data Augmentation (3D)
- EfficientDet Architecture
- Triple IoU Loss
- Inference
- Deployment를 위한 ONNX Conversion Script, Torch Script 추가
- QAT, Grad Clip, SWA, FP16 등 학습 기법 추가 및 테스트