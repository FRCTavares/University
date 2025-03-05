export const config = {
    YOUTUBE_API_KEY: process.env.YOUTUBE_API_KEY || '',
    YOUTUBE_API_BASE_URL: 'https://www.googleapis.com/youtube/v3',
    TRENDING_VIDEOS_ENDPOINT: '/videos',
    MAX_RESULTS: 50,
    REGION_CODE: 'US',
    CHART: 'mostPopular',
};