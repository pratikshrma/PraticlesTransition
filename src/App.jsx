import React from 'react'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import { OrbitControls } from '@react-three/drei'
import ParticleMorphMulti from './Experience'
import Light from './Light'
import { Suspense } from 'react'

const App = () => {
    return (
        <Canvas
            dpr={[1, 2]}
            gl={{
                antialias: true,
                toneMapping: THREE.ACESFilmicToneMapping,
                outputColorSpace: THREE.SRGBColorSpace
            }}
            camera={{ fov: 35, near: 0.1, far: 100, position: [0, 0, 16] }}
            onCreated={({ gl }) => gl.setClearColor("#160920")}
        >
            <Light />
            <Suspense fallback={null}>
                <ParticleMorphMulti
                    // point to multiple models for different states:
                    modelPaths={[
                        "/models/state0.glb",
                        "/models/state1.glb",
                        "/models/state2.glb",
                        "/models/state3.glb"
                    ]}
                    // optionally choose which mesh to use in each file:
                    selectors={[
                        { type: "allMeshes" },
                        { type: "allMeshes" },
                        { type: "allMeshes" },
                        { type: "allMeshes" }
                    ]}
                    autoCycle // automatically morph every few seconds
                    cycleSeconds={4}
                />
                <OrbitControls enableDamping />
            </Suspense>
        </Canvas>
    )
}

export default App