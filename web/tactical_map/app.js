const WORLD_MIN = -40;
const WORLD_MAX = 40;
const WORLD_SIZE = WORLD_MAX - WORLD_MIN;
const ROSBRIDGE_URL = "ws://localhost:9090";

const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const connectionBadge = document.getElementById("connectionBadge");
const modeText = document.getElementById("modeText");
const positionText = document.getElementById("positionText");
const yawText = document.getElementById("yawText");
const targetText = document.getElementById("targetText");
const alertBlock = document.getElementById("alertBlock");
const alertText = document.getElementById("alertText");
const eventLog = document.getElementById("eventLog");

const state = {
  connected: false,
  robot: { x: 0, y: 0, yaw: 0 },
  target: null,
  mode: "IDLE",
  trail: [],
  lastAlert: null,
  lastAlertTime: 0,
};

let socket = null;

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function worldToCanvas(x, y) {
  const rect = canvas.getBoundingClientRect();
  const padding = 28;
  const usableW = rect.width - padding * 2;
  const usableH = rect.height - padding * 2;
  return {
    x: padding + ((x - WORLD_MIN) / WORLD_SIZE) * usableW,
    y: padding + (1 - (y - WORLD_MIN) / WORLD_SIZE) * usableH,
  };
}

function logEvent(text) {
  const item = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  item.textContent = `[${time}] ${text}`;
  eventLog.appendChild(item);
  while (eventLog.children.length > 14) {
    eventLog.removeChild(eventLog.firstChild);
  }
}

function setConnection(connected) {
  state.connected = connected;
  connectionBadge.textContent = connected ? "ROSBRIDGE ONLINE" : "ROSBRIDGE OFFLINE";
  connectionBadge.className = `badge ${connected ? "online" : "offline"}`;
}

function connectRosbridge() {
  socket = new WebSocket(ROSBRIDGE_URL);

  socket.addEventListener("open", () => {
    setConnection(true);
    logEvent("rosbridge connected");
    subscribe("/odom", "nav_msgs/Odometry");
    subscribe("/alerts", "std_msgs/String");
    subscribe("/patrol_state", "std_msgs/String");
    advertise("/mission_command", "std_msgs/String");
  });

  socket.addEventListener("close", () => {
    setConnection(false);
    logEvent("rosbridge disconnected; retrying");
    setTimeout(connectRosbridge, 1600);
  });

  socket.addEventListener("error", () => {
    setConnection(false);
  });

  socket.addEventListener("message", (event) => {
    const packet = JSON.parse(event.data);
    if (packet.op !== "publish") {
      return;
    }
    if (packet.topic === "/odom") {
      handleOdom(packet.msg);
    } else if (packet.topic === "/alerts") {
      handleAlert(packet.msg);
    } else if (packet.topic === "/patrol_state") {
      handlePatrolState(packet.msg);
    }
  });
}

function sendPacket(packet) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    logEvent("rosbridge is not connected");
    return;
  }
  socket.send(JSON.stringify(packet));
}

function subscribe(topic, type) {
  sendPacket({ op: "subscribe", topic, type });
}

function advertise(topic, type) {
  sendPacket({ op: "advertise", topic, type });
}

function publishMission(command) {
  sendPacket({
    op: "publish",
    topic: "/mission_command",
    msg: { data: command },
  });
  logEvent(`mission command: ${command}`);
}

function yawFromQuaternion(q) {
  const sinyCosp = 2 * (q.w * q.z + q.x * q.y);
  const cosyCosp = 1 - 2 * (q.y * q.y + q.z * q.z);
  return Math.atan2(sinyCosp, cosyCosp);
}

function handleOdom(msg) {
  const pose = msg.pose.pose;
  state.robot.x = pose.position.x;
  state.robot.y = pose.position.y;
  state.robot.yaw = yawFromQuaternion(pose.orientation);
  state.trail.push({ x: state.robot.x, y: state.robot.y });
  if (state.trail.length > 160) {
    state.trail.shift();
  }
}

function handleAlert(msg) {
  let summary = msg.data;
  try {
    const payload = JSON.parse(msg.data);
    const confidence = Number(payload.confidence || 0);
    summary = `${payload.event || "person_detected"} confidence=${confidence.toFixed(2)} count=${payload.count || 1}`;
  } catch (error) {
    // Keep raw alert string.
  }
  state.lastAlert = summary;
  state.lastAlertTime = Date.now();
  logEvent(`ALERT ${summary}`);
}

function handlePatrolState(msg) {
  try {
    const payload = JSON.parse(msg.data);
    state.mode = payload.mode || state.mode;
    state.target = payload.target || state.target;
    if (payload.pose) {
      state.robot.x = payload.pose.x;
      state.robot.y = payload.pose.y;
      state.robot.yaw = payload.pose.yaw;
    }
  } catch (error) {
    state.mode = msg.data;
  }
}

function drawGrid(rect) {
  ctx.strokeStyle = "rgba(54, 244, 154, 0.13)";
  ctx.lineWidth = 1;
  for (let meter = WORLD_MIN; meter <= WORLD_MAX; meter += 10) {
    const a = worldToCanvas(meter, WORLD_MIN);
    const b = worldToCanvas(meter, WORLD_MAX);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();

    const c = worldToCanvas(WORLD_MIN, meter);
    const d = worldToCanvas(WORLD_MAX, meter);
    ctx.beginPath();
    ctx.moveTo(c.x, c.y);
    ctx.lineTo(d.x, d.y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(54, 244, 154, 0.42)";
  ctx.strokeRect(28, 28, rect.width - 56, rect.height - 56);
}

function drawWorldLine(points, color, width = 2, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((point, index) => {
    const px = worldToCanvas(point.x, point.y);
    if (index === 0) {
      ctx.moveTo(px.x, px.y);
    } else {
      ctx.lineTo(px.x, px.y);
    }
  });
  ctx.stroke();
  ctx.restore();
}

function drawWorldRect(xMin, yMin, xMax, yMax, fill, stroke) {
  const a = worldToCanvas(xMin, yMax);
  const b = worldToCanvas(xMax, yMin);
  ctx.fillStyle = fill;
  ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
  }
}

function drawMarker(x, y, label, color) {
  const p = worldToCanvas(x, y);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(200, 246, 223, 0.86)";
  ctx.font = "12px monospace";
  ctx.fillText(label, p.x + 9, p.y - 8);
}

function drawMap() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#020706";
  ctx.fillRect(0, 0, rect.width, rect.height);

  drawGrid(rect);
  drawWorldRect(WORLD_MIN, 21, WORLD_MAX, WORLD_MAX, "rgba(30, 104, 142, 0.24)", "rgba(58, 216, 255, 0.22)");
  drawWorldLine([{ x: WORLD_MIN, y: 16 }, { x: WORLD_MAX, y: 16 }], "rgba(230, 198, 75, 0.88)", 3);
  drawWorldLine([{ x: WORLD_MIN + 5, y: 10 }, { x: WORLD_MAX - 5, y: 10 }], "rgba(140, 105, 52, 0.88)", 4, [10, 7]);
  drawWorldLine([{ x: -25, y: 10 }, { x: 25, y: 10 }], "rgba(54, 244, 154, 0.72)", 2);

  drawMarker(-24.8, 10, "Tower W", "rgba(230, 198, 75, 0.95)");
  drawMarker(24.8, 10, "Tower E", "rgba(230, 198, 75, 0.95)");
  drawMarker(-12.8, 8.8, "Bunker", "rgba(150, 180, 150, 0.95)");
  drawMarker(6.4, 7.8, "Bunker", "rgba(150, 180, 150, 0.95)");

  if (state.trail.length > 1) {
    drawWorldLine(state.trail, "rgba(58, 216, 255, 0.52)", 2);
  }

  if (state.target) {
    drawMarker(state.target.x, state.target.y, "Target", "rgba(255, 141, 58, 0.95)");
  }

  const robot = worldToCanvas(state.robot.x, state.robot.y);
  const isAlert = Date.now() - state.lastAlertTime < 4500;
  ctx.save();
  ctx.translate(robot.x, robot.y);
  ctx.rotate(-state.robot.yaw);
  ctx.fillStyle = isAlert ? "#ff3b45" : "#3ad8ff";
  ctx.strokeStyle = "rgba(200, 246, 223, 0.85)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(13, 0);
  ctx.lineTo(-9, -8);
  ctx.lineTo(-6, 0);
  ctx.lineTo(-9, 8);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  if (isAlert) {
    ctx.strokeStyle = "rgba(255, 59, 69, 0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(robot.x, robot.y, 22 + 8 * Math.sin(Date.now() / 160), 0, Math.PI * 2);
    ctx.stroke();
  }
}

function updateText() {
  modeText.textContent = state.mode;
  positionText.textContent = `x ${state.robot.x.toFixed(2)} / y ${state.robot.y.toFixed(2)}`;
  yawText.textContent = `${(state.robot.yaw * 180 / Math.PI).toFixed(1)} deg`;
  targetText.textContent = state.target ? `x ${state.target.x.toFixed(1)} / y ${state.target.y.toFixed(1)}` : "none";

  const alertActive = Date.now() - state.lastAlertTime < 4500;
  alertBlock.classList.toggle("active", alertActive);
  alertText.textContent = alertActive && state.lastAlert ? state.lastAlert : "No active alert";
}

function animate() {
  drawMap();
  updateText();
  requestAnimationFrame(animate);
}

document.getElementById("launchBtn").addEventListener("click", () => publishMission("start_patrol"));
document.getElementById("stopBtn").addEventListener("click", () => publishMission("stop"));
document.getElementById("resumeBtn").addEventListener("click", () => publishMission("resume"));
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
connectRosbridge();
animate();
