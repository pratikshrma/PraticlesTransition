import React from 'react'

const Light = () => {
    return (
        <>
            <ambientLight intensity={1} />
            <directionalLight position={[3, 3, 3]} />
        </>
    )
}

export default Light