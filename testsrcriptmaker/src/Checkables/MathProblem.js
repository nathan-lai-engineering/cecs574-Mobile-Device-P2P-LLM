
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function makeAdditionProblem() {
    const num1 = getRandomInt(1,99);
    const num2 = getRandomInt(1,99);

    return({
        type:'addition', 
        prompt:`what is ${num1} + ${num2}?`, 
        answer:`${num1+num2}`
    })
}
function makeSubtractionProblem() {
    let num1 = getRandomInt(1,99);
    let num2 = getRandomInt(1,99);

    if(num1 < num2) {
        [num1, num2] = [num2, num1];
    }

    return({
        type:'subtraction', 
        prompt:`what is ${num1} - ${num2}?`, 
        answer:`${num1-num2}`
    })
}

export function MakeMathProblem(amount) {

    let generatedMathProblem = [];

    for(let i = 0; i < amount; i++) {
        
        let operation = getRandomInt(0,1)
        if(operation==1) generatedMathProblem.push(makeAdditionProblem(generatedMathProblem));
        if(operation==0) generatedMathProblem.push(makeSubtractionProblem(generatedMathProblem))
    }

    return(generatedMathProblem)
}