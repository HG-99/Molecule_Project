# Molecule Graph Docker Starter

목표는 아래 한 줄을 실제 tensor까지 확인하는 것입니다.

```text
SMILES -> RDKit Mol -> Atom/Bond -> x / edge_index / edge_attr
```

## 1. 빌드

```bash
cd ws_GNN
docker compose build
```

## 2. 컨테이너 생성/실행

```bash
docker compose up -d
```

컨테이너에 들어갑니다.

```bash
docker exec -it mol-encoder-dev bash
```

이미 생성한 컨테이너를 다음 날 다시 사용할 때는:

```bash
docker start mol-encoder-dev
docker exec -it mol-encoder-dev bash
```

## 3. GPU 확인

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

현재 graph 변환 자체는 CPU로 충분하지만, 이후 GINE/D-MPNN 실험으로 바로 확장할 수 있도록 GPU 사용이 가능하게 구성했습니다.

## 4. 첫 실행

기본 예제는 acetic acid (`CC(=O)O`)입니다.

```bash
python src/smiles_to_graph.py
```

다른 molecule:

```bash
python src/smiles_to_graph.py --smiles 'CCO'
python src/smiles_to_graph.py --smiles 'c1ccccc1'
python src/smiles_to_graph.py --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O'
```

명시적 수소 노드를 추가하고 싶으면:

```bash
python src/smiles_to_graph.py --smiles 'CCO' --explicit-h
```

## 5. 출력 tensor

### `x`

shape:

```text
[N_atom, 8]
```

각 행은 원자 하나입니다.

```text
[atomic_num,
 degree,
 formal_charge,
 num_H,
 aromatic,
 in_ring,
 hybridization_id,
 chirality_id]
```

### `edge_index`

shape:

```text
[2, N_directed_edge]
```

예를 들어 `C-C-O`는 화학 결합이 2개지만 message passing을 위해 양방향 edge로 변환하므로:

```text
0 -> 1
1 -> 0
1 -> 2
2 -> 1
```

즉 4개의 directed edge가 됩니다.

### `edge_attr`

shape:

```text
[N_directed_edge, 4]
```

각 행은:

```text
[bond_type_id,
 conjugated,
 in_ring,
 stereo_id]
```

입니다.

## 6. 처음 비교해 볼 SMILES

### Ethanol

```bash
python src/smiles_to_graph.py --smiles 'CCO'
```

### Acetic acid

```bash
python src/smiles_to_graph.py --smiles 'CC(=O)O'
```

single bond와 double bond가 `edge_attr`에서 어떻게 달라지는지 확인합니다.

### Benzene

```bash
python src/smiles_to_graph.py --smiles 'c1ccccc1'
```

`aromatic=1`, `in_ring=1`, `bond_type_id=3`이 어떻게 나타나는지 확인합니다.

### Chiral molecule

```bash
python src/smiles_to_graph.py --smiles 'C[C@H](O)C(=O)O'
```

`chirality_id`가 어떻게 변하는지 확인합니다.

## 7. 다음 구현 단계

```text
Step 1  SMILES -> Graph tensor
Step 2  categorical feature embedding
Step 3  직접 만든 MessagePassing layer
Step 4  GINE baseline
Step 5  graph pooling -> molecule vector
Step 6  ESOL/QM9 prediction
Step 7  local/global/hierarchical encoder
```

중요한 점은 현재 `x`의 숫자를 그대로 continuous feature처럼 Linear에 넣기보다는, 다음 단계에서는 `atomic number`, `hybridization`, `chirality`, `bond type` 같은 categorical column을 각각 Embedding한 뒤 합치는 방식으로 바꾸는 것입니다.
