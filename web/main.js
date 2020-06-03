const num_classes = 256;
const temperature = 0.2;
const ord = c => c.charCodeAt(0);
const chr = i => String.fromCharCode(i);

// js argmax implementation https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
function from_onehot(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function sample_from_logits(logits) {
    return tf.multinomial(logits.map(x => x/temperature), 1).arraySync()[0];
}

const txtLyrics = document.getElementById("txtLyrics");
const btnGenerate = document.getElementById("btnGenerate");
let rnn_model

async function run_demo() {
    // seed text
    s = txtLyrics.value.split('');
    if (s.length === 0) {
        alert("Enter some starting text");
        return;
    }
    
    for (let i = 0; i < 500; ++i) {
        // Wrap the input in a tensor
        let input = tf.tensor1d(s.map(ord));
        // Expand batch dimension and predict
        let res = rnn_model.predict(input.expandDims(0));
        let output = await res.array();

        let out_chr = chr(sample_from_logits(output[0]));
        
        s.push(out_chr);
        txtLyrics.value = s.join("");
    }
} 

function onload(model) {
    model.summary();
    rnn_model = model;
    btnGenerate.value = "Generate";
    btnGenerate.disabled = false;
    btnGenerate.onclick = run_demo;
}

tf.loadLayersModel('./rnn_weights/model.json').then(model => onload(model));
