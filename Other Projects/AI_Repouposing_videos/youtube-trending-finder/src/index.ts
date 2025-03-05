import { TrendingService } from './services/trendingService';
import { config } from './config/config';
import { logger } from './utils/logger';

const initializeApp = async () => {
    try {
        logger.info('Initializing YouTube Trending Finder...');
        
        const trendingService = new TrendingService();
        const trendingVideos = await trendingService.getTrendingVideos();

        logger.info('Fetched trending videos:', trendingVideos);
    } catch (error) {
        logger.error('Error initializing application:', error);
    }
};

initializeApp();