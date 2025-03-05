import axios from 'axios';
import { Video } from '../models/video';
import { config } from '../config/config';

export class YouTubeClient {
    private apiKey: string;
    private baseUrl: string;

    constructor() {
        this.apiKey = config.youtubeApiKey;
        this.baseUrl = 'https://www.googleapis.com/youtube/v3';
    }

    public async fetchTrendingVideos(): Promise<Video[]> {
        try {
            const response = await axios.get(`${this.baseUrl}/videos`, {
                params: {
                    part: 'snippet,statistics',
                    chart: 'mostPopular',
                    regionCode: 'US',
                    maxResults: 10,
                    key: this.apiKey,
                    publishedAfter: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
                }
            });

            return response.data.items.map((item: any) => new Video(
                item.id,
                item.snippet.title,
                item.snippet.description,
                item.statistics.viewCount
            ));
        } catch (error) {
            throw new Error(`Failed to fetch trending videos: ${error.message}`);
        }
    }
}