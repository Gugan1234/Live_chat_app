import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;

    const res = await fetch('http://localhost:5000/send_message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();
    setMessages(prev => [...prev, ...data]);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="chat-header">Mini RAG Application</div>
      {/* <div className="chat-messages" id="chatMessages">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`chat-message ${msg.sender}-message`}
          >
            <div className="message-text">{msg.text}</div>
            <div className="timestamp">{msg.time}</div>
          </div>
        ))}
      </div> */}
      <div className="chat-messages" id="chatMessages">
  {messages.map((msg, idx) => (
   <div className={`chat-message-wrapper ${msg.sender}`}>
  <img src={msg.avatar} alt="avatar" className="avatar" />
  <div className={`chat-message ${msg.sender}-message`}>
    <div className="message-text">{msg.text}</div>
    <div className="timestamp">{msg.time}</div>
  </div>
</div>

  ))}
</div>
      <div className="chat-input">
        <input
          type="text"
          id="messageInput"
          placeholder="Type your message..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;