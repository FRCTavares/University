import YouTubeClient from '../api/youtubeClient';
import Video from '../models/video';

export default class TrendingService {
    private youtubeClient: YouTubeClient;

    constructor() {
        this.youtubeClient = new YouTubeClient();
    }

    public async getTrendingVideos(): Promise<Video[]> {
        const trendingData = await this.youtubeClient.fetchTrendingVideos();
        return this.processTrendingData(trendingData);
    }

    private processTrendingData(data: any): Video[] {
        return data.items.map((item: any) => {
            return new Video({
                id: item.id,
                title: item.snippet.title,
                description: item.snippet.description,
                viewCount: item.statistics.viewCount,
            });
        });
    }
}