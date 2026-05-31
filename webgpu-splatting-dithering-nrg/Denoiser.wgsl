// Prostorski bilateralni filter z globinsko edge-stopping utežjo
// Inspiriran z [1] Mara et al. HPG 2017 — Bilateral Spatial Filter
//
// Za vsak piksel p vzorčimo 7×7 okno sosedov q.
// Skupna utež w(p,q) = prostorska_utež x globinska_utež
// Izhod = Σ w(p,q)·C(q) / Σ w(p,q)

@group(0) @binding(0) var colorTexture: texture_2d<f32>;
@group(0) @binding(1) var depthTexture: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

struct Uniforms {
    // sigma_s: prostorska širina Gaussove uteži (v pikslih)
    // večji sigma_s -> bolj glajenje, bolj zamegljeno
    sigmaS: f32,
    // sigma_d: globinska širina edge-stopping uteži
    // večji sigma_d -> filter meša čez globinske robove (manj edge-preserving)
    // manjši sigma_d -> filter strogo loči po globini (ostrejši robovi)
    sigmaD: f32,
}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Full-screen trikotnik — en trikotnik pokrije cel zaslon
// Učinkovitejše kot quad (2 trikotnika)
const vertices = array(
    vec2f(-1, -1),
    vec2f( 3, -1),
    vec2f(-1,  3),
);

struct VertexInput {
    @builtin(vertex_index) index: u32,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) texcoords: vec2f,
}

struct FragmentInput {
    @location(0) texcoords: vec2f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clipPosition = vec4f(vertices[input.index], 0, 1);
    // texcoords: [0,1]×[0,1], Y obrnjen (WebGPU ima Y navzdol v teksturah)
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;
    return output;
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    // Dimenzije teksture za računanje odmikov v pikslih
    let texSize = vec2f(textureDimensions(colorTexture));
    // texelSize: razdalja med sosednjima piksloma v teksturnih koordinatah
    let texelSize = 1.0 / texSize;

    // Pikselne koordinate centralnega piksla p
    let centerCoord = input.texcoords;

    // Vzorčimo barvo in globino centralnega piksla
    let centerColor = textureSample(colorTexture, texSampler, centerCoord).rgb;
    let centerDepth = textureSample(depthTexture, texSampler, centerCoord).r;

    // Akumulatorji za tehtano vsoto
    var weightedColorSum = vec3f(0.0);
    var weightSum = 0.0;

    // 7×7 okno: odmiki od -3 do +3 v vsaki osi
    // Skupaj 49 vzorcev na piksel
    for (var dy: i32 = -3; dy <= 3; dy++) {
        for (var dx: i32 = -3; dx <= 3; dx++) {

            // Teksturne koordinate soseda q
            let offset = vec2f(f32(dx), f32(dy)) * texelSize;
            let sampleCoord = centerCoord + offset;

            // Vzorčimo barvo in globino soseda
            let sampleColor = textureSample(colorTexture, texSampler, sampleCoord).rgb;
            let sampleDepth = textureSample(depthTexture, texSampler, sampleCoord).r;

            // --- Prostorska utež ---
            // Gaussova funkcija prostorske razdalje (v pikslih)
            // ||p-q||² = dx²+dy² (v pikslih)
            let spatialDist2 = f32(dx*dx + dy*dy);
            let wSpatial = exp(-spatialDist2 / (2.0 * uniforms.sigmaS * uniforms.sigmaS));

            // --- Globinska utež (edge-stopping) ---
            // Samo kadar imata OBA piksla geometrijo (depth > 0), uporabimo globinsko
            // razdaljo za zaustavljanje filtra na robovih.
            //
            // Posebnost stohastičnega renderiranja: zavrženi fragmenti imajo depth = 0.0
            // (clearValue), ker discard ne zapiše v color attachment. To je ENAKO kot
            // resnično ozadje — ne moremo razlikovati med:
            //   (a) ozadje (ni geometrije)
            //   (b) stohastično zavržen fragment (je geometrija, ampak ta okvir ni bil narisan)
            //
            // Rešitev: kadar je vsaj eden depth=0, ne kaznujemo -> wDepth=1.0.
            // To dovoli zavrženim pikslom, da se povprečijo s sosednjimi veljavnimi piksli
            // in s tem odpravijo stohastični šum. Pravi globinski edge-stopping deluje
            // samo med dvema foreground piksloma (oba depth > 0).
            var wDepth = 1.0;
            if (centerDepth > 0.0 && sampleDepth > 0.0) {
                // Oba sta foreground -> globinski edge-stopping
                let depthDiff = centerDepth - sampleDepth;
                wDepth = exp(-(depthDiff * depthDiff) / (2.0 * uniforms.sigmaD * uniforms.sigmaD));
            }
            // Sicer (eden ali oba depth=0): wDepth = 1.0 -> prostorska utež ureja prispevek

            // --- Firefly suppression (iz [1] Mara et al.) ---
            // Če je vzorec drastično svetlejši od centra, ga stisnemo navzdol.
            // Preprečuje da en izjemno svetel piksel (firefly) pokvari povprečje.
            // Luminanca po ITU-R BT.709: L = 0.2126R + 0.7152G + 0.0722B
            let centerLum = dot(centerColor, vec3f(0.2126, 0.7152, 0.0722));
            let sampleLum = dot(sampleColor, vec3f(0.2126, 0.7152, 0.0722));
            var clampedSample = sampleColor;
            let threshold = 4.0 * centerLum + 0.01; // +0.01 da ne delimo z 0 v temnih regijah
            if (sampleLum > threshold) {
                clampedSample = sampleColor * (threshold / sampleLum);
            }

            // Skupna utež
            let w = wSpatial * wDepth;

            weightedColorSum += w * clampedSample;
            weightSum += w;
        }
    }

    // Normaliziramo — delimo z vsoto uteži
    // Če je weightSum = 0 (izjemen primer), vrnemo originalno barvo
    if (weightSum > 0.0) {
        output.color = vec4f(weightedColorSum / weightSum, 1.0);
    } else {
        output.color = vec4f(centerColor, 1.0);
    }

    return output;
}
