import React, { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import { FiSend } from 'react-icons/fi';
import { Button } from '../ui/button';

const ChatInput = ({ onSendMessage, disabled, quickActions }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleQuickAction = (action) => {
    if (!disabled) {
      onSendMessage(action);
    }
  };

  return (
    <div className="bg-surface pt-2">
      {quickActions?.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2 px-1">
          {quickActions.map((action) => (
            <Button
              key={action}
              variant="outline"
              size="sm"
              onClick={() => handleQuickAction(action)}
              disabled={disabled}
              className="text-xs rounded-full h-7"
            >
              {action}
            </Button>
          ))}
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="relative flex items-center"
      >
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder="AI asistana bir soru sorun..."
          disabled={disabled}
          className="flex-1 bg-bg-tertiary border-2 border-transparent rounded-lg pl-4 pr-12 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-medical-blue transition-colors disabled:opacity-60"
          rows={1}
          maxLength={500}
        />
        <Button
          type="submit"
          disabled={disabled || !message.trim()}
          size="icon"
          className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg"
          aria-label="Mesaj gönder"
        >
          <FiSend size={16} />
        </Button>
      </form>
    </div>
  );
};

ChatInput.propTypes = {
  onSendMessage: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  quickActions: PropTypes.arrayOf(PropTypes.string),
};

export default ChatInput;
