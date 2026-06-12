# Radiance — Bible Worlds · Unified 3DGS Asset PoC

하나의 canonical **3DGS** 자산(Peter / 베드로)으로 **relight · edit · compose** 를 모두
입증하는 PoC. 자세한 미션·파이프라인·실행 토폴로지는 작업 노트인 **[`radiance.org`](./radiance.org)**
가 source of truth 입니다. 이 README는 **RunPod 환경 셋업과 launcher 실행 방법**만 다룹니다.

---

## 실행 토폴로지 (요약)

```
[로컬]                                          [원격 RunPod GPU Pod]
Claude Code / Emacs ──SSH 터널(8888)──▶ JupyterLab ──▶ r3dg 커널
       org-babel(:kernel r3dg)                         (학습/COLMAP/분해 실행)
```

- 코드 편집·git·Claude Code·Emacs = **전부 로컬**. **코드 실행만** 원격 Pod 커널에서.
- Pod의 모든 영속 상태(venv·repo·data·outputs)는 **Network Volume(`/workspace`)** 에 보존
  → Pod를 지웠다 같은 볼륨으로 다시 만들면 재컴파일 없이 몇 분 내 재개.

---

## 사전 준비

1. **uv 설치** (로컬). launcher는 uv로 실행하며, 의존성(`runpod`, `python-dotenv`)은
   스크립트 안에 PEP 723 인라인 메타데이터로 선언되어 `uv run`이 자동으로 가져옵니다.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **API 키**. `.env.runpod.local` 파일 생성 (git에서 제외됨):
   ```dotenv
   RUNPOD_API_KEY=rpa_xxx...
   # (선택) Pod 부팅 시 canonical repo를 클론하려면:
   # RADIANCE_REPO_URL=git@github.com:you/radiance.git
   ```
3. **SSH 공개키**. `~/.ssh/id_ed25519.pub`(또는 `id_rsa.pub`)가 있으면 자동으로 Pod에
   주입되어 SSH 터널링이 가능해집니다.

---

## GPU 선택

| GPU | 아키텍처 | VRAM | 비고 |
|-----|----------|------|------|
| **RTX 3090** (기본값) | Ampere `sm_86` | 24GB | 원 repo(Relightable3DGaussian) 레퍼런스 GPU. 저렴. |
| RTX A6000 | Ampere `sm_86` | 48GB | BVH ray tracing / compose에서 VRAM 부족 시 업그레이드. |

> ⚠️ 스택이 **PyTorch 1.12.1 + cu116** 에 고정되어 있고, 1.12는 Ampere(`sm_80/86`)까지만
> 커널을 내장합니다. **RTX 4090/L40(Ada `sm_89`), H100(Hopper `sm_90`)은 지원되지 않으니
> 사용하지 마세요.** (A100 `sm_80`, A6000/A5000/A4500/3090 `sm_86` 은 OK)

> 🔑 **반드시 SECURE cloud 를 쓰세요 (launcher 기본값).** emacs-jupyter 는 `ssh -L` 터널이
> 필수인데, COMMUNITY cloud 의 exposed-TCP 포트는 컨테이너 sshd 로 라우팅되지 않아(호스트
> sshd 가 응답·거부) SSH 가 실패합니다. SECURE cloud 는 전용 public IP 로 컨테이너에 직접
> 닿습니다. 단, SECURE 는 GPU 재고가 더 적습니다 — 3090 이 없으면 A4500(20GB)/A100(80GB) 등
> Ampere 호환 GPU 로 대체하세요. (`--gpu`/DC 가용성은 launcher 로 조회)

---

## Launcher 사용법

모든 명령은 `uv run`으로 실행합니다 (별도 설치 불필요).

### 0. API 연결 테스트
```bash
uv run scripts/runpod_launcher.py --test-api
```

### 1. Network Volume 생성 (최초 1회)
영속 저장소. venv·repo·data·outputs·checkpoints가 모두 여기 보존됩니다.
```bash
uv run scripts/runpod_launcher.py --create-volume \
    --volume-name radiance-vol --volume-size 100 --datacenter EU-RO-1
```
```bash
uv run scripts/runpod_launcher.py --list-volumes      # volume id 확인
```
> Pod는 볼륨과 **같은 datacenter**에 떠야 합니다. launcher가 볼륨의 DC를 자동으로 맞춥니다.

### 2. Pod 생성
```bash
uv run scripts/runpod_launcher.py --create \
    --gpu "NVIDIA GeForce RTX 3090" \
    --volume-id <vol_id>
```
**repo 클론 URL은 자동 감지**됩니다 — 우선순위: `--repo-url` > `RADIANCE_REPO_URL` >
로컬 `git remote get-url origin`. SSH 형식(`git@github.com:...`)이어도 pod 클론용으로는
**HTTPS로 자동 변환**(공개 repo는 인증 없이 clone/pull). 즉 이 repo에 GitHub remote만
걸어두면 이후 `--create` 시 알아서 `/workspace/radiance`로 클론됩니다.

부팅 시 자동으로 수행되는 작업 (`radiance.org` §1.1~1.4 미러링):
1. apt 시스템 의존성 + `sshd` 기동 (+ SSH 공개키 주입)
2. **canonical repo 클론/pull** → `/workspace/radiance` (위 자동 감지 URL)
3. 외부 repo 클론 → `/workspace/radiance/third_party/`
   (Relightable3DGaussian, gaussian-grouping, nvdiffrast)
4. **uv venv 빌드** `/workspace/envs/r3dg` — uv-managed Python(볼륨)+ PyTorch 1.12.1+cu116
   + `torch_scatter`/`kornia` + CUDA 확장 컴파일(simple-knn, bvh, r3dg-rasterization, nvdiffrast)
   + colmap/ffmpeg(apt)
5. **`r3dg` 커널 등록** (emacs-jupyter `:kernel r3dg` 대상)
6. **JupyterLab** `:8888` 기동

> 첫 부팅은 venv 빌드+CUDA 컴파일로 **15~30분** 소요됩니다. 같은 볼륨으로 재생성 시엔
> 컴파일 결과가 보존되어 빠르게 재개됩니다. 진행 상황: Pod에 SSH로 들어가
> `tail -f /workspace/startup.log`.

빠른 반복을 위해 venv 빌드를 건너뛰고 수동으로 만들려면 `--skip-bootstrap` (repo는 그대로 클론됨).

### 3. 상태 확인 (SSH 터널 + Jupyter URL 출력)
```bash
uv run scripts/runpod_launcher.py --status
```
출력 예시의 안내대로:
```bash
# 1) 로컬에서 SSH 터널
ssh -L 8888:localhost:8888 root@<public_ip> -p <ssh_port>
# 2) emacs-jupyter / 브라우저로 접속
#    http://localhost:8888/lab?token=radiance_token   (kernel: r3dg)
```
이제 org-babel `jupyter-python` 블록(`:kernel r3dg`)이 원격 커널에서 실행됩니다.

> ⚠️ private key(`~/.ssh/id_ed25519`)에 passphrase가 있으면 **먼저 agent에 올리세요**:
> `ssh-add ~/.ssh/id_ed25519` (passphrase 1회). 안 그러면 키는 맞아도 비대화형 접속이 거부됩니다.

### 3-1. `~/.ssh/config` 자동 기록 — `ssh radiance-pod` 한 줄로
Pod의 public IP/포트는 **stop/start 마다 바뀝니다.** launcher가 현재 엔드포인트를
`~/.ssh/config`의 관리 블록(`Host radiance-pod`)에 써줍니다(터널 8888 포함). `--create`/`--start`
시 자동 실행되며, 수동 갱신도 가능:
```bash
uv run scripts/runpod_launcher.py --write-ssh-config   # 현재 IP/포트로 ~/.ssh/config 갱신
ssh radiance-pod                                        # 접속 + 8888 터널 자동
```
(전용 `~/.ssh/known_hosts.radiance` 사용 → IP 교체로 인한 host-key 충돌 방지.)

### 4. 기타
```bash
uv run scripts/runpod_launcher.py --list             # 내 Pod 목록
uv run scripts/runpod_launcher.py --stop             # Pod 정지(볼륨 유지, GPU 반납)
uv run scripts/runpod_launcher.py --start            # 정지된 Pod 재개 + ~/.ssh/config 갱신
uv run scripts/runpod_launcher.py --delete           # Pod 종료(볼륨 유지)
uv run scripts/runpod_launcher.py --delete-volume <vol_id>   # 볼륨 삭제(주의: 영구)
```
Pod 이름 기본값은 `radiance-<username>`. `--name`으로 변경 가능.

---

## 재시작 후 이어가기

- **`--stop` 했던 경우**: `--start`로 재개. (`--create`는 "이미 존재"로 거부됨)
- **`--delete` 했던 경우**: **같은 Network Volume**으로 `--create --volume-id <id>`. venv·CUDA
  확장·repo·data가 볼륨에 남아 있어 빌드 단계를 건너뜁니다.

둘 다:
- public IP/포트가 **새로 배정**되므로 `--start`/`--create`가 `~/.ssh/config`를 자동 갱신합니다
  (수동: `--write-ssh-config`). 이후 `ssh radiance-pod`.
- jupyter는 부팅 후 **~3\~6분 뒤 자동 재기동**(apt 재설치 + 커널 등록). `ssh radiance-pod
  'tail -f /workspace/startup.log'`로 `[JL] Starting JupyterLab` 확인.
- ⚠️ GPU 재고가 'Low'면 **재개 실패 가능**(stop은 GPU 반납). 짧은 휴식은 켜둔 채 두는 게 안전.

(자세한 RESUME 절차는 `radiance.org` §6)

---

## 비용 절약

- 작업이 끝나면 `--stop` (볼륨 과금만, GPU 과금 정지) 또는 `--delete`.
- 인터랙티브/시각 작업은 Xesktop, 배치 컴퓨트만 RunPod로 분리 (`radiance.org` "인프라 매핑").

---

> 참고: `radiance.org` §1.2는 원래 conda 기반 부트스트랩을 기술합니다. 이 launcher는 동일한
> 패키지 셋을 **uv**로 설치하도록 구현되어 있습니다(방법은 swappable, 결과 동일).
