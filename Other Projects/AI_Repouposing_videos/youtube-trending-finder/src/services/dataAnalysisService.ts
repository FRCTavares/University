export class DataAnalysisService {
    calculateAverageViews(videos: Array<{ viewCount: number }>): number {
        const totalViews = videos.reduce((sum, video) => sum + video.viewCount, 0);
        return totalViews / videos.length || 0;
    }

    identifyTrends(videos: Array<{ title: string; viewCount: number }>): Array<{ title: string; trend: string }> {
        const trends = videos.map(video => {
            let trend = 'Stable';
            if (video.viewCount > 10000) {
                trend = 'Trending';
            } else if (video.viewCount < 1000) {
                trend = 'Declining';
            }
            return { title: video.title, trend };
        });
        return trends;
    }
}