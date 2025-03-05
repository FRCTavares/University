YouTube Trending Finder
A Node.js application that discovers the most popular videos on YouTube within the last 24 hours, sorted by various metrics.

Features
Retrieves trending videos from YouTube using the YouTube Data API
Filters videos published in the last 24 hours
Sorts videos by different popularity metrics (views, likes, comments)
Supports filtering by category/genre
Exports results in various formats (JSON, CSV)
Command-line interface for easy usage
Installation

# Clone the repository
git clone https://github.com/yourusername/youtube-trending-finder.git

# Navigate to the project directory
cd youtube-trending-finder

# Install dependencies
npm install

Configuration
Obtain a YouTube Data API key from the Google Cloud Console
Create a .env file in the root directory based on .env.example:
Add your API key to the .env file:
Usage
Basic usage
Advanced options
API Reference
This project uses the YouTube Data API v3. For more information about the API, visit YouTube Data API Documentation.

Requirements
Node.js 14.x or higher
A valid YouTube Data API key
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.

Disclaimer
This project is not affiliated with, maintained, authorized, endorsed, or sponsored by YouTube or Google.