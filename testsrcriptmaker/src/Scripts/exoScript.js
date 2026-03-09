
export function exoScript(text) {

    let exoScript = '';

    let textArr = text.split('\n');

    textArr.forEach(line => {

        let prompt = '';
        try {
            let obj = JSON.parse(line);
            prompt=obj.prompt;
        } catch (e) {
            prompt=line;
        }


        exoScript+=`curl -N -X POST http://localhost:52415/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    "model": "mlx-community/Qwen3-0.6B-8bit",
    "messages": [
      {"role": "user", "content": "${prompt}"}
    ],
    "stream": true
  } >> output.txt;`;
    });

    return exoScript
}