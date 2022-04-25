import "./style.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as dat from "dat.gui";

const deepcopy = (obj) => JSON.parse(JSON.stringify(obj));

// Canvas
const canvas = document.querySelector("canvas.webgl");

// Scene
const scene = new THREE.Scene();

const axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// Objects
const geometry = new THREE.SphereGeometry(1, 60, 30);
geometry.rotateX(Math.PI / 2);

// Mesh
const sphere = new THREE.Mesh(
  geometry,
  new THREE.MeshNormalMaterial({ flatShading: true })
);

const wireframe = new THREE.Mesh(
  geometry,
  new THREE.MeshBasicMaterial({
    color: 0x000000,
    flatShading: true,
    wireframe: true,
    wireframeLinewidth: 2,
  })
);

const group = new THREE.Group();

group.add(sphere);
group.add(wireframe);

scene.add(group);

/**
 * Sizes
 */
const sizes = {
  width: window.innerWidth,
  height: window.innerHeight,
};

// Camera
const camera = new THREE.PerspectiveCamera(
  75,
  sizes.width / sizes.height,
  0.1,
  100
);

camera.position.y = -2.5;
camera.position.z = 0.7;
camera.lookAt(geometry.center());

scene.add(camera);

// Controls
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// Renderer
const renderer = new THREE.WebGLRenderer({ canvas: canvas });

renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));

/* Animate */

const pos = deepcopy(geometry.attributes.position.array);

// GUI

const gui = new dat.GUI();
const params = { t: 0, rotation: { x: 0, y: 0, z: 0 } };
gui.add(params, "t", 0, 1 - 0.0001, 0.0001).listen();

/*
const rotFolder = gui.addFolder("Rotation");
rotFolder.add(params.rotation, "x", 0, 2 * Math.PI, 0.001).listen();
rotFolder.add(params.rotation, "y", 0, 2 * Math.PI, 0.001).listen();
rotFolder.add(params.rotation, "z", 0, 2 * Math.PI, 0.001).listen();

const posFolder = gui.addFolder("Position");
posFolder.add(camera.position, "x", -5, 5, 0.001).listen();
posFolder.add(camera.position, "y", -5, 5, 0.001).listen();
posFolder.add(camera.position, "z", -5, 5, 0.001).listen();
posFolder.open();
*/

const tick = () => {
  geometry.rotateY(Math.PI / 2);

  controls.update();

  sphere.rotation.x = params.rotation.x;
  sphere.rotation.y = params.rotation.y;
  sphere.rotation.z = params.rotation.z;

  wireframe.rotation.x = params.rotation.x;
  wireframe.rotation.y = params.rotation.y;
  wireframe.rotation.z = params.rotation.z;

  for (let i = 0; i < geometry.attributes.position.count; i++) {
    const t = params.t;

    const x = pos[i * 3];
    const y = pos[i * 3 + 1];
    const z = pos[i * 3 + 2];

    const m = 2 / (2 * (1 - t) + (1 - z) * t);

    // set new position
    geometry.attributes.position.setX(i, m * x);
    geometry.attributes.position.setY(i, m * y);
    geometry.attributes.position.setZ(i, -t + z * (1 - t));
  }

  geometry.computeVertexNormals();
  geometry.attributes.position.needsUpdate = true;

  renderer.render(scene, camera);
  window.requestAnimationFrame(tick);
};

window.addEventListener("resize", () => {
  // Update sizes
  sizes.width = window.innerWidth;
  sizes.height = window.innerHeight;

  // Update camera
  camera.aspect = sizes.width / sizes.height;
  camera.updateProjectionMatrix();

  // Update renderer
  renderer.setSize(sizes.width, sizes.height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
});

tick();
