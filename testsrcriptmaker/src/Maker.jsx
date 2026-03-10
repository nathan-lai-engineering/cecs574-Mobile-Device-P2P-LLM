import {InputLabel, MenuItem, Select, Box, TextField, Button, FormControl} from '@mui/material';
import { useState } from 'react'
import {MakeMathProblem} from './Checkables/MathProblem'
import { exoScript } from './Scripts/exoScript';


export function Maker({prompt, setPrompt}) {

    const [type, setType] = useState('Math Problem')
    const [number, setNumber] = useState(3)

    const [scriptType, setScriptType] = useState('');
    const [script, setScript] = useState('');

    function handleMakeExoScript() {
        setScriptType("exo")
        setScript(exoScript(prompt))
    }
    function handleMakeLLinkScript() {
        setScriptType("lingua")

    }
    function handleMakeLLamaScript() {
        setScriptType("llama")

    }

    function handleGenerate() {
        if(type === 'Math Problem') {  
            let problems = MakeMathProblem(number)
            
            let newPrompt = prompt;

            for(let i in problems) {
                console.log(problems[i])
                newPrompt+=JSON.stringify(problems[i]);
                newPrompt+='\n'
            }
            

            setPrompt(newPrompt)
        }
    }

    function handleNumberChange(e) {
        let numberStr = e.target.value;
        if(numberStr.length == 0) numberStr='1';
        setNumber(parseInt(numberStr))
    }

    return<>
        <Box display={'flex'}>
            <TextField label="Amount" variant="outlined" sx={{mr:1,width:'8em'}} size="small" value={number} onChange={handleNumberChange}/>

            <FormControl sx={{width:'16em', mr:1}} size='small' >
                <InputLabel>Prompt type</InputLabel>
                <Select
                    value={type}
                    label="Prompt type"
                    onChange={(e)=>{setType(e.target.value)}}
                >
                    <MenuItem value={'Math Problem'}>Math Problem</MenuItem>
                    <MenuItem value={'Letter Counting'}>Letter Counting</MenuItem>
                </Select>
            </FormControl>
            <Button variant="contained" sx={{width:'12em'}} onClick={handleGenerate} >Make Problem</Button>

            
        </Box>

        <TextField label='Answer key' sx={{width:'36em', mt:3}} value={prompt} onChange={(_e,v)=>{setPrompt(v)}} multiline minRows={8} maxRows={16}></TextField>
        
        <Box display={'flex'} sx={{mt:6}}>
            <Button variant={scriptType === "exo" ? "contained" : 'outlined'} sx={{width:'12em', mr:1}} onClick={handleMakeExoScript} size='small'> Exo </Button>
            <Button variant={scriptType === "lingua" ? "contained" : 'outlined'} sx={{width:'12em', mr:1}} onClick={handleMakeLLinkScript} size='small'> Lingualinked </Button>
            <Button variant={scriptType === "llama" ? "contained" : 'outlined'} sx={{width:'12em'}} onClick={handleMakeLLamaScript} size='small'> LLAMA </Button>
        </Box>

        <TextField label='Test script' sx={{width:'36em', mt:3}} value={script} onChange={(_e,v)=>{setScript(v)}} multiline minRows={8} maxRows={16}></TextField>
        
    </>
}