// src/ParticleMorphMulti.jsx
import React, { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useThree, useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { MeshSurfaceSampler } from "three/examples/jsm/math/MeshSurfaceSampler.js";
import gsap from "gsap";

import particlesVertexShader from "./shaders/particles/vertex.glsl";
import particlesFragmentShader from "./shaders/particles/fragment.glsl";

/* =========================
   Mesh & sampling utilities
   ========================= */

// Collect meshes that have a position attribute, honoring a selector.
// Supported selectors: {type:"allMeshes"} (default), {type:"firstMesh"},
// {type:"index", value:n}, {type:"name", value:"MeshName"}
function collectMeshesFromGLTF(gltf, selector) {
    const meshes = [];
    gltf.scene.traverse((o) => {
        if (o.isMesh && o.geometry?.attributes?.position) meshes.push(o);
    });
    if (!meshes.length) return [];

    const sel = selector || { type: "allMeshes" };

    if (sel.type === "firstMesh") return [meshes[0]];
    if (sel.type === "index") {
        const idx = Math.max(0, Math.min(sel.value ?? 0, meshes.length - 1));
        return [meshes[idx]];
    }
    if (sel.type === "name") {
        const found = meshes.find((m) => m.name === sel.value);
        return [found || meshes[0]];
    }
    // default: all meshes
    return meshes;
}

// Merge vertex positions (no transforms applied) into a single attribute
function mergePositions(meshes) {
    let total = 0;
    for (const m of meshes) total += m.geometry.attributes.position.count;
    const out = new Float32Array(total * 3);
    let offset = 0;
    for (const m of meshes) {
        const src = m.geometry.attributes.position.array;
        out.set(src, offset);
        offset += src.length;
    }
    return new THREE.Float32BufferAttribute(out, 3);
}

// Bake a mesh's world transform into a cloned geometry (positions + normals)
function bakeMeshToWorld(mesh) {
    mesh.updateWorldMatrix(true, false);
    const g = mesh.geometry.clone();

    // Ensure we have a fresh copy of attributes
    const pos = g.getAttribute("position").array;
    const m4 = mesh.matrixWorld.clone();
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.length; i += 3) {
        v.set(pos[i + 0], pos[i + 1], pos[i + 2]).applyMatrix4(m4);
        pos[i + 0] = v.x;
        pos[i + 1] = v.y;
        pos[i + 2] = v.z;
    }
    g.attributes.position.needsUpdate = true;

    if (g.getAttribute("normal")) {
        const nrm = g.getAttribute("normal").array;
        const n = new THREE.Vector3();
        const nm3 = new THREE.Matrix3().getNormalMatrix(m4);
        for (let i = 0; i < nrm.length; i += 3) {
            n.set(nrm[i + 0], nrm[i + 1], nrm[i + 2]).applyMatrix3(nm3).normalize();
            nrm[i + 0] = n.x; nrm[i + 1] = n.y; nrm[i + 2] = n.z;
        }
        g.attributes.normal.needsUpdate = true;
    }

    // Return a baked mesh at identity (sampler will now be in "world space")
    const baked = new THREE.Mesh(g);
    baked.updateWorldMatrix(true, false);
    return baked;
}

// Compute loose bbox size of multiple baked meshes
function getMergedBboxSize(meshes) {
    const box = new THREE.Box3();
    const tmp = new THREE.Box3();
    for (const m of meshes) {
        m.geometry.computeBoundingBox();
        tmp.copy(m.geometry.boundingBox);
        box.union(tmp);
    }
    const size = new THREE.Vector3();
    box.getSize(size);
    return { box, size };
}

// Uniformly normalize baked meshes to fit a target size and be centered
function normalizeBakedMeshes(meshes, targetSize = 4) {
    if (!meshes.length) return;
    const { box, size } = getMergedBboxSize(meshes);
    const maxAxis = Math.max(size.x, size.y, size.z) || 1;
    const scale = targetSize / maxAxis;

    const center = new THREE.Vector3();
    box.getCenter(center);

    for (const m of meshes) {
        const posAttr = m.geometry.getAttribute("position");
        const arr = posAttr.array;
        for (let i = 0; i < arr.length; i += 3) {
            arr[i + 0] = (arr[i + 0] - center.x) * scale;
            arr[i + 1] = (arr[i + 1] - center.y) * scale;
            arr[i + 2] = (arr[i + 2] - center.z) * scale;
        }
        posAttr.needsUpdate = true;
        m.geometry.computeBoundingSphere();
    }
}

// Build a MeshSurfaceSampler for each baked mesh; skip zero-area meshes
function buildSamplers(bakedMeshes) {
    const samplers = [];
    for (const bm of bakedMeshes) {
        const g = bm.geometry;
        // Estimate triangle count; skip degenerate/empty
        const triCount = g.index ? g.index.count / 3 : g.getAttribute("position").count / 3;
        if (triCount <= 0) continue;
        try {
            samplers.push(new MeshSurfaceSampler(bm).build());
        } catch (e) {
            // If the sampler fails (non-triangles / bad geometry), skip
            // console.warn("Sampler build failed for a mesh:", e);
        }
    }
    return samplers;
}

// Sample N points from a list of meshes (robust):
// 1) bake to world
// 2) normalize (optional)
// 3) build samplers
// 4) if none available, fallback to vertex merge (and upsample)
function samplePointsFromMeshes(meshes, count, { normalize = true, targetSize = 4 } = {}) {
    const baked = meshes.map(bakeMeshToWorld);
    if (normalize) normalizeBakedMeshes(baked, targetSize);

    const samplers = buildSamplers(baked);
    const out = new Float32Array(count * 3);

    if (samplers.length === 0) {
        // Fallback: merge vertices (already baked & normalized), then upsample
        let total = 0;
        for (const b of baked) total += b.geometry.getAttribute("position").count;
        if (total === 0) return new THREE.Float32BufferAttribute(out, 3);

        const merged = new Float32Array(total * 3);
        let off = 0;
        for (const b of baked) {
            const arr = b.geometry.getAttribute("position").array;
            merged.set(arr, off);
            off += arr.length;
        }
        const srcCount = merged.length / 3;
        for (let i = 0; i < count; i++) {
            const s3 = (Math.floor(Math.random() * srcCount)) * 3;
            const i3 = i * 3;
            out[i3 + 0] = merged[s3 + 0];
            out[i3 + 1] = merged[s3 + 1];
            out[i3 + 2] = merged[s3 + 2];
        }
        return new THREE.Float32BufferAttribute(out, 3);
    }

    // Round-robin across samplers (simple and effective)
    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
        const s = samplers[i % samplers.length];
        s.sample(p, n); // baked mesh → already in world-ish coords (post-normalization)
        const i3 = i * 3;
        out[i3 + 0] = p.x;
        out[i3 + 1] = p.y;
        out[i3 + 2] = p.z;
    }
    return new THREE.Float32BufferAttribute(out, 3);
}

/* ================
   Main component
   ================ */

export default function ParticleMorphMulti({
    modelPaths = [],
    selectors = [],
    autoCycle = false,
    cycleSeconds = 4,
    // Toggle dense clouds:
    useSurfaceSampling = true,
    particleCount = 40000,
    normalizeModels = true,     // helps when some models seem to "disappear"
    normalizeTargetSize = 4,    // overall size after normalization
    // Visual tweaks
    colorA = "#ff7300",
    colorB = "#0091ff",
    baseSize = 0.35
}) {
    const { size } = useThree();
    const dpr = Math.min(window.devicePixelRatio, 2);

    // Load GLBs (with Draco)
    const gltfs = useLoader(
        GLTFLoader,
        modelPaths,
        (loader) => {
            const draco = new DRACOLoader();
            draco.setDecoderPath("/draco/");
            loader.setDRACOLoader(draco);
        }
    );

    // Build per-state position attributes
    const stateAttributes = useMemo(() => {
        if (!gltfs?.length) return [];

        if (useSurfaceSampling) {
            // Robust surface sampling per GLB
            return gltfs.map((gltf, i) => {
                const meshes = collectMeshesFromGLTF(gltf, selectors[i]);
                return samplePointsFromMeshes(meshes, particleCount, {
                    normalize: normalizeModels,
                    targetSize: normalizeTargetSize
                });
            });
        } else {
            // Vertex path (merge positions → will be padded later)
            return gltfs
                .map((gltf, i) => {
                    const meshes = collectMeshesFromGLTF(gltf, selectors[i]);
                    return mergePositions(meshes);
                })
                .filter(Boolean);
        }
    }, [gltfs, selectors, useSurfaceSampling, particleCount, normalizeModels, normalizeTargetSize]);

    // Build geometry/material/API
    const { geometry, material, api } = useMemo(() => {
        if (!stateAttributes.length) return {};

        let padded = stateAttributes;

        // Vertex mode may require padding to longest count; surface mode doesn't
        if (!useSurfaceSampling) {
            let maxCount = 0;
            for (const a of stateAttributes) maxCount = Math.max(maxCount, a.count);

            padded = stateAttributes.map((a) => {
                if (a.count === maxCount) return a;
                const out = new Float32Array(maxCount * 3);
                const src = a.array;
                for (let i = 0; i < maxCount; i++) {
                    const dst3 = i * 3;
                    if (dst3 < src.length) {
                        out[dst3 + 0] = src[dst3 + 0];
                        out[dst3 + 1] = src[dst3 + 1];
                        out[dst3 + 2] = src[dst3 + 2];
                    } else {
                        const s3 = Math.floor(a.count * Math.random()) * 3;
                        out[dst3 + 0] = src[s3 + 0];
                        out[dst3 + 1] = src[s3 + 1];
                        out[dst3 + 2] = src[s3 + 2];
                    }
                }
                return new THREE.Float32BufferAttribute(out, 3);
            });
        }

        // Geometry
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", padded[0]);
        geom.setAttribute("aPositionTarget", padded[1] ?? padded[0]);

        // Per-point sizes
        const count = geom.getAttribute("position").count;
        const sizes = new Float32Array(count);
        for (let i = 0; i < count; i++) sizes[i] = Math.random();
        geom.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));

        // Material (same shader)
        const uniforms = {
            uSize: new THREE.Uniform(baseSize),
            uResolution: new THREE.Uniform(new THREE.Vector2(size.width * dpr, size.height * dpr)),
            uProgress: new THREE.Uniform(0),
            uColorA: new THREE.Uniform(new THREE.Color(colorA)),
            uColorB: new THREE.Uniform(new THREE.Color(colorB))
        };
        const mat = new THREE.ShaderMaterial({
            vertexShader: particlesVertexShader,
            fragmentShader: particlesFragmentShader,
            uniforms,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            transparent: true
        });

        // Morph API
        let currentIndex = 0;
        const api = {
            count: padded.length,
            morphTo: (idx, { duration = 3, ease = "linear" } = {}) => {
                if (!padded[idx]) return;
                geom.setAttribute("position", padded[currentIndex]);
                geom.setAttribute("aPositionTarget", padded[idx]);
                geom.attributes.position.needsUpdate = true;
                geom.attributes.aPositionTarget.needsUpdate = true;
                uniforms.uProgress.value = 0;
                gsap.to(uniforms.uProgress, { value: 1, duration, ease });
                currentIndex = idx;
            },
            setColors: (a, b) => { uniforms.uColorA.value.set(a); uniforms.uColorB.value.set(b); },
            setSize: (s) => { uniforms.uSize.value = s; },
            updateResolution: (w, h, d) => { uniforms.uResolution.value.set(w * d, h * d); }
        };

        return { geometry: geom, material: mat, api };
    }, [stateAttributes, size.width, size.height, dpr, baseSize, colorA, colorB, useSurfaceSampling]);

    // Keep resolution uniform fresh
    useEffect(() => {
        if (!material) return;
        material.uniforms.uResolution.value.set(size.width * dpr, size.height * dpr);
    }, [size, dpr, material]);

    // Quick buttons to trigger states
    const buttonsRef = useRef(null);
    useEffect(() => {
        if (!api) return;
        const wrap = document.createElement("div");
        wrap.style.cssText = "position:fixed;left:12px;top:12px;display:flex;gap:8px;z-index:10;font-family:ui-sans-serif;";
        for (let i = 0; i < api.count; i++) {
            const b = document.createElement("button");
            b.textContent = `state ${i}`;
            b.style.cssText = "padding:6px 10px;border-radius:10px;border:1px solid #fff2;background:#ffffff18;color:#fff;cursor:pointer;";
            b.onclick = () => api.morphTo(i);
            wrap.appendChild(b);
        }
        document.body.appendChild(wrap);
        buttonsRef.current = wrap;
        return () => { if (buttonsRef.current) document.body.removeChild(buttonsRef.current); };
    }, [api]);

    // Auto cycle
    useEffect(() => {
        if (!api || !autoCycle || api.count < 2) return;
        let idx = 1;
        const id = setInterval(() => {
            api.morphTo(idx % api.count, { duration: 3 });
            idx++;
        }, Math.max(1, cycleSeconds) * 1000);
        return () => clearInterval(id);
    }, [api, autoCycle, cycleSeconds]);

    if (!geometry || !material) return null;

    return (
        <points frustumCulled={false} scale={[2.5, 2.5, 2.5]}>
            <primitive object={geometry} attach="geometry" />
            <primitive object={material} attach="material" />
        </points>
    );
}
