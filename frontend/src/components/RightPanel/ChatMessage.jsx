import React from 'react';
import PropTypes from 'prop-types';
import { FiUser, FiCpu, FiExternalLink } from 'react-icons/fi';
import { cn } from '../../lib/utils';

const ChatMessage = ({ message }) => {
  const isAI = message.sender === 'ai';

  const Avatar = () => (
    <div className={cn(
      "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
      isAI ? 'bg-status-success/10' : 'bg-medical-blue/10'
    )}>
      {isAI ? (
        <FiCpu className="text-status-success" size={16} />
      ) : (
        <FiUser className="text-medical-blue" size={16} />
      )}
    </div>
  );

  return (
    <div className={cn(
      "flex items-start gap-3 animate-fade-in",
      !isAI && 'flex-row-reverse'
    )}>
      <Avatar />
      <div className={cn(
        "w-full max-w-[80%]",
        !isAI && "flex flex-col items-end"
      )}>
        <div className={cn(
          "rounded-lg px-3.5 py-2.5",
          isAI
            ? 'bg-bg-tertiary'
            : 'bg-medical-blue text-white'
        )}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
        {isAI && message.source && (
          <a
            href={message.source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-text-tertiary mt-2 hover:text-text-accent transition-colors"
          >
            <FiExternalLink size={12} />
            <span>Kaynak: {message.source.title}</span>
          </a>
        )}
        <p className="text-xs text-text-tertiary mt-1.5">
          {new Date(message.timestamp).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>
    </div>
  );
};

ChatMessage.propTypes = {
  message: PropTypes.shape({
    content: PropTypes.string.isRequired,
    sender: PropTypes.oneOf(['ai', 'user']).isRequired,
    timestamp: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    source: PropTypes.shape({
      title: PropTypes.string,
      url: PropTypes.string,
    }),
  }).isRequired,
};

export default ChatMessage;

