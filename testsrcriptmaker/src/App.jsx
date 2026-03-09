import { useState } from 'react'
import {Tabs, Tab, Box} from '@mui/material';
import { Maker } from './Maker';

function App() {
  
  const [tab, setTab] = useState(0);
  const [prompt, setPrompt] = useState('');



  return (
    <Box display={'grid'} justifyItems={'center'} sx={{width:'100vw'}}>
      <Tabs value={tab} onChange={(_e,v)=>{setTab(v)}} sx={{mb:3}}>
        <Tab label="Prompt maker" />
        <Tab label="Result parser" />
      </Tabs>

      {tab === 0 && <Maker prompt={prompt} setPrompt={setPrompt}/>}
      {tab === 1 && <>TODO</>}

    </Box>
  )
}

export default App
