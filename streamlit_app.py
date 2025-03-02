import streamlit as st
import streamlit.components.v1 as components

# HTML & JavaScript for a 3D Hand Animation using Three.js
threejs_code = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
    <style>
        body { margin: 0; }
        canvas { display: block; }
    </style>
</head>
<body>
    <script>
        // Create Scene
        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        var renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Create Hand (Simple Cube for demo)
        var geometry = new THREE.BoxGeometry(1, 2, 0.5);
        var material = new THREE.MeshBasicMaterial({color: 0xffcc00});
        var hand = new THREE.Mesh(geometry, material);
        scene.add(hand);

        camera.position.z = 5;

        function animate() {
            requestAnimationFrame(animate);
            hand.rotation.x += 0.05;
            hand.rotation.y += 0.05;
            renderer.render(scene, camera);
        }

        animate();
    </script>
</body>
</html>
"""

# Streamlit UI
st.title("South African Sign Language (SASL) - 3D Avatar Signing 'Hello'")

# Embed Three.js animation inside Streamlit
components.html(threejs_code, height=500)

st.write("This 3D avatar represents a waving hand as the 'Hello' sign.")
