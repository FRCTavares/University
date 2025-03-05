import { createLogger, format, transports } from 'winston';

const logger = createLogger({
    level: 'info',
    format: format.combine(
        format.timestamp(),
        format.printf(({ timestamp, level, message }) => {
            return `${timestamp} [${level}]: ${message}`;
        })
    ),
    transports: [
        new transports.Console(),
        new transports.File({ filename: 'error.log', level: 'error' }),
        new transports.File({ filename: 'combined.log' })
    ],
});

export const logInfo = (message: string) => {
    logger.info(message);
};

export const logWarning = (message: string) => {
    logger.warn(message);
};

export const logError = (message: string) => {
    logger.error(message);
};