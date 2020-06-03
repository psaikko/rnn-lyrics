const num_classes = 256
const temperature = 0.2
const ord = c => c.charCodeAt(0);
const chr = i => String.fromCharCode(i);

// js argmax implementation https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
function from_onehot(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function sample_from_logits(logits) {
    return tf.multinomial(logits.map(x => x/temperature), 1).arraySync()[0]
}

const outputDiv = document.getElementById("outputDiv");

async function run_demo(model) {
    model.summary()

    // seed text
    s = "Who let the dogs".split()

    for (let i = 0; i < 500; ++i) {
        // Wrap the input in a tensor
        let input = tf.tensor1d(s.map(ord));
        // Expand batch dimension and predict
        let res = model.predict(input.expandDims(0));
        let output = await res.array();

        let out_chr = chr(sample_from_logits(output[0]));
        
        s.push(out_chr);
        outputDiv.innerText = s.join("");
    }
} 

tf.loadLayersModel('./rnn_weights/model.json').then(run_demo);
