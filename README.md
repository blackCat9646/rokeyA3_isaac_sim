# DMZ Sentry

Isaac Sim과 ROS 2 Humble을 이용한 4족 보행 정찰 로봇 시뮬레이션 프로젝트입니다.  
ANYmal이 DMZ 스타일의 울타리, 강가, 벙커, 감시탑이 있는 환경을 순찰하고, 카메라 기반 YOLO 사람 감지, Nav2 기반 순찰, 웹 전술 지도, 관측용 줌 카메라를 함께 사용합니다.

## 현재 구현된 기능

- Isaac Sim standalone 시뮬레이션
- ANYmal 기반 4족 보행 로봇
- GP 스타일 지형, 울타리, 강, 벙커, 감시탑, 경고 표지, 조명
- 움직이는 사람 target 시나리오
- RGB-D 감지 카메라
- 별도 관측용 Inspector 카메라
- YOLOv8 사람 감지
- `/alerts` 기반 경보 발생
- Nav2 기반 waypoint 순찰
- 웹 전술 지도
- 웹에서 출격, 홈, 정지, 재개 명령
- 웹에서 target 클릭 시 Inspector 카메라가 해당 target을 바라봄
- 웹에서 Inspector 카메라 pan/tilt/zoom 수동 조작
- target 확인 처리: Clear 버튼을 누르면 Confirmed 초록색 상태로 변경

## 전체 구조

```text
Isaac Sim
  ANYmal
  SentryFrontCamera          YOLO 감지용 카메라
  SentryInspectionCamera     target 확인/줌 관측용 카메라
  IntruderScenario           움직이는 사람 target
  /camera/image_raw
  /camera/depth
  /inspection_camera/image_raw
  /odom
  /tf

ROS 2
  yolo_person_detector       사람 감지, /alerts 발행
  inspection_bridge          웹 명령과 Isaac Sim 파일 브리지 연결
  Nav2                       경로 계획과 순찰 주행
  nav2_patrol_controller     웹 mission command를 Nav2 goal로 변환
  rosbridge                  웹과 ROS 2 연결

Web
  tactical_map               전술 지도, target 표시, 출격/정지/카메라 제어
```

## Target 위치 표시 방식

현재 웹에 표시되는 target 위치는 Isaac Sim 내부의 시뮬레이션 좌표를 사용합니다.

```text
Isaac Sim IntruderScenario
→ /tmp/dmz_sentry_intruder_states.json
→ inspection_bridge
→ /intruder_states
→ web tactical map
```

즉 지금은 **YOLO + Depth로 실제 위치를 추정한 방식이 아니라**, 시뮬레이션이 알고 있는 ground-truth 좌표를 웹에 표시합니다.  
YOLO는 target 표시를 위한 위치 계산보다는 사람 감지와 alert 발생에 사용됩니다.

추후 현실적인 방식으로 확장하려면 다음 구조로 바꿀 수 있습니다.

```text
YOLO bbox
+ /camera/depth
+ /camera/camera_info
+ TF
→ world 좌표 추정
→ /tracked_targets publish
```

또는 3D LiDAR를 사용할 경우:

```text
YOLO bbox
+ LiDAR point cloud projection
→ bbox 안 point cloud cluster
→ target 3D 위치 추정
```

## 웹 전술 지도 기능

웹 지도는 `web/tactical_map`에 있습니다.

기능:

- 로봇 현재 위치 표시
- 순찰 waypoint 표시
- target 위치 표시
- YOLO alert 상태 표시
- target 클릭 시 Inspector 카메라가 해당 target을 바라봄
- Confirmed target은 초록색으로 표시
- target이 재소환되어 위치가 크게 바뀌면 Confirmed 상태 자동 해제

버튼:

- `출격`: 순찰 시작
- `홈`: 홈 위치로 복귀
- `정지`: 정지
- `재개`: 이전 순찰 모드 재개
- `Pan Left / Pan Right`: Inspector 카메라 좌우 조작
- `Tilt Up / Tilt Down`: Inspector 카메라 상하 조작
- `Center`: Inspector 카메라 정면 복귀
- `Zoom + / Zoom -`: Inspector 카메라 줌 인/아웃
- `Reset`: 줌 초기화
- `Clear`: target 추적 해제, 선택 target을 Confirmed 상태로 변경

## Inspector 카메라

기존 카메라는 YOLO 감지용으로 계속 넓게 앞을 봅니다.  
Inspector 카메라는 target 확인용으로 따로 추가된 관측 카메라입니다.

```text
SentryFrontCamera
  YOLO 감지용
  /camera/image_raw
  /camera/annotated

SentryInspectionCamera
  target 확인/줌 관측용
  /inspection_camera/image_raw
```

웹에서 target을 클릭하면 `/inspection_camera/command`가 발행되고, `inspection_bridge`가 이 명령을 Isaac Sim에 전달합니다. Isaac Sim은 해당 좌표를 바라보도록 Inspector 카메라 방향을 갱신합니다.

현재 Inspector 카메라는 실제 물리 짐벌 모델이 아니라, 코드로 카메라 방향을 바꾸는 **가상 짐벌** 방식입니다.

## YOLO 사람 감지

현재 사용하는 학습 모델:

```text
models/dmz_person_calibration_001_best.pt
```

YOLO 노드:

```text
ros2_ws/src/dmz_sentry_perception/dmz_sentry_perception/yolo_person_detector.py
```

출력 토픽:

- `/detections_text`: 감지 결과 JSON
- `/alerts`: confidence 기준 이상이면 alert 발행
- `/camera/annotated`: bbox가 그려진 확인용 이미지

학습 데이터 변환 스크립트:

```text
scripts/convert_replicator_to_yolo.py
```

학습은 별도 Python 코드가 아니라 Ultralytics CLI로 수행했습니다.

```bash
yolo detect train \
  model=yolov8n.pt \
  data=/home/rokey/dev_ws/dmz_sentry/datasets/yolo_person_calibration_001/data.yaml \
  epochs=50 \
  imgsz=640 \
  device=0 \
  project=/home/rokey/dev_ws/dmz_sentry/runs/yolo \
  name=dmz_person_calibration_001
```

## Nav2 순찰

현재 순찰은 Nav2 기반입니다.

```text
web 출격 버튼
→ /mission_command
→ nav2_patrol_controller
→ /navigate_to_pose action
→ Nav2
→ /cmd_vel_nav2_raw
→ cmd_vel_safety_filter
→ /cmd_vel
→ Isaac Sim ANYmal
```

이 프로젝트에서는 SLAM을 아직 사용하지 않습니다.  
현재는 DMZ 환경을 알고 있다고 가정하고, 정적 map과 `world` frame을 이용하는 known-map 방식입니다.

Nav2 확인용 명령:

```bash
ros2 action list | grep navigate_to_pose
ros2 topic echo /patrol_state
ros2 topic echo /cmd_vel
```

`/navigate_to_pose`가 없으면 Nav2가 켜지지 않은 상태입니다.

## 자주 확인하는 토픽

```bash
ros2 topic hz /camera/image_raw
ros2 topic hz /camera/annotated
ros2 topic hz /inspection_camera/image_raw
ros2 topic echo /intruder_states --once
ros2 topic echo /alerts --once
ros2 topic echo /patrol_state
ros2 topic info /inspection_camera/command
```

정상 상태 예:

```text
/camera/image_raw             average rate ...
/inspection_camera/image_raw  average rate ...
/intruder_states              data: "{...}"
/inspection_camera/command    Publisher count 1, Subscription count 1
/navigate_to_pose             action 존재
```

## 폴더 구조

```text
dmz_sentry/
  isaacsim/
    anymal_gp_terrain.py              Isaac Sim 메인 시뮬레이션

  ros2_ws/src/dmz_sentry_perception/
    yolo_person_detector.py           YOLO 사람 감지 노드

  ros2_ws/src/dmz_sentry_control/
    nav2_patrol_controller.py         Nav2 순찰 컨트롤러
    cmd_vel_safety_filter.py          ANYmal 안정 주행용 속도 필터
    inspection_bridge.py              웹/ROS/Isaac Sim 카메라 명령 브리지
    config/nav2_dmz_params.yaml       Nav2 설정
    maps/                             정적 지도

  web/tactical_map/
    index.html
    app.js
    style.css                         웹 전술 지도

  scripts/
    demo_dmz_sim.sh
    demo_yolo_detector.sh
    demo_inspection_bridge.sh
    demo_nav2_bringup.sh
    demo_nav2_patrol_controller.sh
    demo_rosbridge.sh
    demo_tactical_map.sh

  models/
    dmz_person_calibration_001_best.pt
```

## 빌드

ROS 2 노드를 수정했거나 처음 실행하는 경우:

```bash
cd /home/rokey/dev_ws/dmz_sentry/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## 실행 커맨드 정리

아래 순서대로 터미널을 열어서 실행하면 현재 데모 전체가 동작합니다.

```bash
# Terminal 1: Isaac Sim 시뮬레이션
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_dmz_sim.sh
```

```bash
# Terminal 2: Inspector 카메라/target 위치 브리지
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_inspection_bridge.sh
```

```bash
# Terminal 3: YOLO 사람 감지
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_yolo_detector.sh
```

```bash
# Terminal 4: Nav2 실행
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_nav2_bringup.sh
```

```bash
# Terminal 5: Nav2 순찰 컨트롤러
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_nav2_patrol_controller.sh
```

```bash
# Terminal 6: rosbridge websocket
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_rosbridge.sh
```

```bash
# Terminal 7: 웹 전술 지도
cd /home/rokey/dev_ws/dmz_sentry
./scripts/demo_tactical_map.sh
```

웹 브라우저에서 아래 주소를 엽니다.

```text
http://localhost:8080
```

rqt로 카메라를 확인하려면:

```bash
rqt_image_view
```

감지 카메라:

```text
/camera/annotated
```

Inspector 카메라:

```text
/inspection_camera/image_raw
```
