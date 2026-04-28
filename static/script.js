```js
/**
 * Real-time Dashboard Updates Module
 * 
 * This module handles real-time updates for the cryptocurrency trading bot dashboard.
 * It supports both WebSocket connections (preferred) and HTTP polling as fallback.
 * Provides real-time market data, trading signals, and alerts.
 * 
 * @module DashboardUpdates
 */

// Configuration
const CONFIG = {
  // WebSocket endpoint (update with your actual endpoint)
  WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
  
  // Polling interval in milliseconds (fallback when WebSocket is unavailable)
  POLLING_INTERVAL: 5000,
  
  // API endpoints
  API_BASE_URL: '/api',
  ENDPOINTS: {
    MARKET_DATA: '/market-data',
    SIGNALS: '/signals',
    ALERTS: '/alerts',
    PORTFOLIO: '/portfolio',
    STATUS: '/status'
  },
  
  // Reconnection settings
  RECONNECT_DELAY: 3000,
  MAX_RECONNECT_ATTEMPTS: 10,
  
  // Update throttling
  THROTTLE_INTERVAL: 1000
};

/**
 * Dashboard Update Manager
 * Handles all real-time data updates for the dashboard
 */
class DashboardUpdateManager {
  constructor() {
    this.ws = null;
    this.pollingInterval = null;
    this.reconnectAttempts = 0;
    this.isConnected = false;
    this.updateCallbacks = new Map();
    this.throttledUpdates = new Map();
    
    // Initialize the connection
    this.initialize();
  }
  
  /**
   * Initialize the dashboard updates
   * Attempts WebSocket connection first, falls back to polling
   */
  initialize() {
    if (this.isWebSocketSupported()) {
      this.connectWebSocket();
    } else {
      console.warn('WebSocket not supported, falling back to HTTP polling');
      this.startPolling();
    }
    
    // Set up periodic health checks
    this.healthCheckInterval = setInterval(() => this.healthCheck(), 30000);
  }
  
  /**
   * Check if WebSocket is supported by the browser
   * @returns {boolean}
   */
  isWebSocketSupported() {
    return 'WebSocket' in window && typeof WebSocket === 'function';
  }
  
  /**
   * Connect to WebSocket server
   */
  connectWebSocket() {
    try {
      this.ws = new WebSocket(CONFIG.WS_URL);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected successfully');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.stopPolling();
        this.emit('connection', { status: 'connected', type: 'websocket' });
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', { type: 'websocket', message: error.message || 'Unknown error' });
      };
      
      this.ws.onclose = (event) => {
        console.log(`WebSocket closed (code: ${event.code})`);
        this.isConnected = false;
        this.emit('connection', { status: 'disconnected', type: 'websocket' });
        
        // Attempt reconnection
        if (this.reconnectAttempts < CONFIG.MAX_RECONNECT_ATTEMPTS) {
          this.reconnectAttempts++;
          setTimeout(() => {
            console.log(`Reconnection attempt ${this.reconnectAttempts}/${CONFIG.MAX_RECONNECT_ATTEMPTS}`);
            this.connectWebSocket();
          }, CONFIG.RECONNECT_DELAY * Math.min(this.reconnectAttempts, 5));
        } else {
          console.warn('Max reconnection attempts reached, switching to polling');
          this.startPolling();
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.startPolling();
    }
  }
  
  /**
   * Handle incoming WebSocket messages
   * @param {Object} data - Parsed message data
   */
  handleMessage(data) {
    if (!data || !data.type) {
      console.warn('Invalid message received:', data);
      return;
    }
    
    switch (data.type) {
      case 'market_data':
        this.throttledEmit('marketData', data.payload);
        break;
      case 'signal':
        this.emit('signal', data.payload);
        break;
      case 'alert':
        this.emit('alert', data.payload);
        break;
      case 'portfolio':
        this.throttledEmit('portfolio', data.payload);
        break;
      case 'status':
        this.emit('status', data.payload);
        break;
      case 'error':
        console.error('Server error:', data.message);
        this.emit('error', { type: 'server', message: data.message });
        break;
      default:
        console.warn('Unknown message type:', data.type);
    }
  }
  
  /**
   * Start HTTP polling as fallback
   */
  startPolling() {
    if (this.pollingInterval) return;
    
    console.log('Starting HTTP polling');
    this.pollingInterval = setInterval(() => {
      this.pollAllEndpoints();
    }, CONFIG.POLLING_INTERVAL);
    
    // Initial poll
    this.pollAllEndpoints();
    
    this.emit('connection', { status: 'connected', type: 'polling' });
  }
  
  /**
   * Stop HTTP polling
   */
  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
      console.log('HTTP polling stopped');
    }
  }
  
  /**
   * Poll all API endpoints for updates
   */
  async pollAllEndpoints() {
    const endpoints = Object.values(CONFIG.ENDPOINTS);
    
    try {
      const results = await Promise.allSettled(
        endpoints.map(endpoint => this.fetchData(endpoint))
      );
      
      results.forEach((result, index) => {
        if (result.status === 'fulfilled' && result.value) {
          const endpoint = endpoints[index];
          const type = endpoint.replace('/', '');
          this.handlePolledData(type, result.value);
        } else if (result.status === 'rejected') {
          console.error(`Polling failed for ${endpoints[index]}:`, result.reason);
        }
      });
    } catch (error) {
      console.error('Error during polling:', error);
    }
  }
  
  /**
   * Fetch data from API endpoint with error handling
   * @param {string} endpoint - API endpoint path
   * @returns {Promise<Object|null>}
   */
  async fetchData(endpoint) {
    try {
      const response = await fetch(`${CONFIG.API_BASE_URL}${endpoint}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      return null;
    }
  }
  
  /**
   * Handle polled data and emit appropriate events
   * @param {string} type - Data type
   * @param {Object} data - Polled data
   */
  handlePolledData(type, data) {
    if (!data || !data.success) return;
    
    switch (type) {
      case 'market-data':
        if (data.marketData) {
          this.throttledEmit('marketData', data.marketData);
        }
        break;
      case 'signals':
        if (data.signals) {
          data.signals.forEach(signal => this.emit('signal', signal));
        }
        break;
      case 'alerts':
        if (data.alerts) {
          data.alerts.forEach(alert => this.emit('alert', alert));
        }
        break;
      case 'portfolio':
        if (data.portfolio) {
          this.throttledEmit('portfolio', data.portfolio);
        }
        break;
      case 'status':
        if (data.status) {
          this.emit('status', data.status);
        }
        break;
    }
  }
  
  /**
   * Throttle emissions to prevent UI overload
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  throttledEmit(event, data) {
    const now = Date.now();
    const lastUpdate = this.throttledUpdates.get(event) || 0;
    
    if (now - lastUpdate >= CONFIG.THROTTLE_INTERVAL) {
      this.throttledUpdates.set(event, now);
      this.emit(event, data);
    }
  }
  
  /**
   * Emit event to registered callbacks
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  emit(event, data) {
    const callbacks = this.updateCallbacks.get(event) || [];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in callback for event "${event}":`, error);
      }
    });
    
    // Also dispatch custom DOM event for non-JS listeners
    const customEvent = new CustomEvent(`dashboard:${event}`, { detail: data });
    window.dispatchEvent(customEvent);
  }
  
  /**
   * Register callback for specific event
   * @param {string} event - Event name to listen for
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (typeof callback !== 'function') {
      console.error('Callback must be a function');
      return;
    }
    
    if (!this.updateCallbacks.has(event)) {
      this.updateCallbacks.set(event, []);
    }
    
    this.updateCallbacks.get(event).push(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = this.updateCallbacks.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }
  
  /**
   * Remove all callbacks for specific event or all events
   * @param {string} [event] - Optional event name to clear
   */
  off(event) {
    if (event) {
      this.updateCallbacks.delete(event);
    } else {
      this.updateCallbacks.clear();
    }
  }
  
  /**
   * Health check to ensure connection is alive
   */
  healthCheck() {
    if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      try {
        // Send ping frame or message
        if (this.ws.ping) {
          this.ws.ping();
        } else {
          // Fallback to sending a simple message
          const pingMessage = JSON.stringify({ type: 'ping', timestamp: Date.now() });
          if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(pingMessage);
          }
        }
      } catch (error) {
        console.error('Health check failed:', error);
      }
    } else if (!this.isConnected && !this.pollingInterval) {
      // If not connected and not polling, try to reconnect or start polling
      if (this.isWebSocketSupported()) {
        this.connectWebSocket();
      } else {
        this.startPolling();
      }
    }
  }
  
  /**
   * Send message through WebSocket
   * @param {Object} message - Message to send
   */
  send(message) {
    if (!message || typeof message !== 'object') {
      console.error('Invalid message to send');
      return false;
    }
    
    if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      try {
        const serialized = JSON.stringify(message);
        this.ws.send(serialized);
        return true;
      } catch (error) {
        console.error('Error sending message:', error);
        return false;
      }
    } else {
      console.warn('Cannot send message: WebSocket not connected');
      return false;
    }
  }
  
  /**
   * Clean up resources and disconnect
   */
  destroy() {
    // Clear intervals
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    
    // Close WebSocket
    if (this.ws) {
      try {
        this.ws.close(1000, 'Client disconnecting');
      } catch (error) {
        console.error('Error closing WebSocket:', error);
      }
      this.ws = null;
    }
    
    // Clear all callbacks and throttled updates
    this.updateCallbacks.clear();
    this.throttledUpdates.clear();
    
    console.log('Dashboard update manager destroyed');
  }
}

/**
 * UI Update Functions
 * These functions update the DOM with real-time data
 */

/**
 * Update market data display on dashboard
 * @param {Object} marketData - Market data object
 */
function updateMarketData(marketData) {
  const container = document.getElementById('market-data-container');
  if (!container || !marketData) return;
  
  try {
    // Update price displays
    Object.entries(marketData).forEach(([symbol, data]) => {
      const element = document.getElementById(`price-${symbol}`);
      if (element && data.price !== undefined) {
        element.textContent = formatPrice(data.price);
        
        // Add color coding for price changes
        if (data.change24h !== undefined) {
          element.className = `price ${data.change24h >= 0 ? 'up' : 'down'}`;
          
          // Update change indicator
          const changeElement = document.getElementById(`change-${symbol}`);
          if (changeElement) {
            changeElement.textContent = `${data.change24h >= 0 ? '+' : ''}${data.change24h.toFixed(2)}%`;
            changeElement.className = `change ${data.change24h >= 0 ? 'positive' : 'negative'}`;
          }
          
          // Update volume display
          const volumeElement = document.getElementById(`volume-${symbol}`);
          if (volumeElement && data.volume !== undefined) {
            volumeElement.textContent = formatVolume(data.volume);
          }
          
          // Update high/low prices
          const highElement = document.getElementById(`high-${symbol}`);
          const lowElement = document.getElementById(`low-${symbol}`);
          
          if (highElement && data.high !== undefined) highElement.textContent = formatPrice(data.high);
          if (lowElement && data.low !== undefined) lowElement.textContent = formatPrice(data.low);
          
          // Update last update timestamp
          const timestampElement = document.getElementById(`timestamp-${symbol}`);
          if (timestampElement && data.timestamp !== undefined) {
            timestampElement.textContent = new Date(data.timestamp).toLocaleTimeString();
            timestampElement.title = new Date(data.timestamp).toLocaleString();
          }
          
          // Update sparkline chart if available and Chart.js is loaded
          if (window.sparklineCharts && window.sparklineCharts[symbol] && Array.isArray(data.history)) {
            window.sparklineCharts[symbol].data.datasets[0].data = data.history.slice(-50);
            window.sparklineCharts[symbol].update('none');
          }
          
          // Trigger custom event for other components
          const event = new CustomEvent('market-update', { 
            detail: { symbol, ...data },
            bubbles: true 
          });
          element.dispatchEvent(event);
          
          // Update last updated timestamp in footer or status bar
          updateLastUpdated(symbol, Date.now());
          
          // Check for significant price movements and trigger alerts if needed
          checkPriceAlert(symbol, data.price, data.change24h);
          
          // Update technical indicators display if available
          updateTechnicalIndicators(symbol, data);
          
          // Update order book display if available
          updateOrderBook(symbol, data.orderBook || null);
          
          // Update recent trades display if available  
          updateRecentTrades(symbol, data.recentTrades || []);
          
          // Update market depth chart if available and Chart.js is loaded  
          updateMarketDepth(symbol, data.marketDepth || null);
          
          // Update funding rate display for perpetual contracts  
          updateFundingRate(symbol, data.fundingRate || null); 
          
          // Update open interest display for futures contracts  
          updateOpenInterest(symbol, data.openInterest || null); 
          
          // Update liquidation levels display  
          updateLiquidationLevels(symbol, data.liquidationLevels || null); 
          
          // Update market indicators display  
          updateMarketIndicators(symbol, data.marketIndicators || null); 
          
          // Update market sentiment display  
          updateMarketSentiment(symbol, data.marketSentiment || null); 
          
          // Update market news display  
          updateMarketNews(symbol, data.marketNews || []); 
          
          // Update market events display  
          updateMarketEvents(symbol, data.marketEvents || []); 
          
          // Update market announcements display  
          updateMarketAnnouncements(symbol, data.marketAnnouncements || []); 
          
          // Update market warnings display  
          updateMarketWarnings(symbol, data.marketWarnings || []); 
          
          // Update market errors display  
          updateMarketErrors(symbol, data.marketErrors || []); 
          
          // Update market logs display  
          updateMarketLogs(symbol, data.marketLogs || []); 
          
          // Update market metrics display  
          updateMarketMetrics(symbol, data.marketMetrics || null); 
          
          // Update market statistics display  
          updateMarketStatistics(symbol, data.marketStatistics || null); 
          
          // Update market summary display  
          updateMarketSummary(symbol, data.marketSummary || null); 
          
          // Update market overview display  
          updateMarketOverview(symbol, data.marketOverview || null); 
          
          // Update market details display  
          updateMarketDetails(symbol, data.marketDetails || null); 
          
          // Update market information display  
          updateMarketInformation(symbol, data.marketInformation || null); 
          
          // Update market status display  
          updateMarketStatus(symbol, data.marketStatus || null); 
          
            } else {  
              console.warn(`Missing required fields for symbol ${symbol}`);  
            }  
            
            // Additional validation for nested objects  
            validateMarketDataNestedFields(data);  
            
            } catch (error) {  
              console.error(`Error updating market data for ${symbol}:`, error);  
            }  
            
            });  
            
            } catch (error) {  
              console.error('Error updating market data container:', error);  
            }  

/**
* Validate nested fields in market data object  
* @param {Object} data - Market data object to validate  
*/  
function validateMarketDataNestedFields(data) {  

// Validate technical indicators structure  
if (data.technicalIndicators && typeof data.technicalIndicators === 'object') {  

// Validate RSI value range  
if (data.technicalIndicators.rsi !== undefined &&   
(typeof data.technicalIndicators.rsi !== 'number' ||   
isNaN(data.technicalIndicators.rsi))) {  

console.warn(`Invalid RSI value: ${data.technicalIndicators.rsi}`);  

}  

// Validate MACD structure  
if (data.technicalIndicators.macd && typeof data.technicalIndicators.macd === 'object') {  

if (!Array.isArray(data.technicalIndicators.macd.line)) {  

console.warn('MACD line must be an array');  

}  

if (!Array.isArray(data.technicalIndicators.macd.signal)) {  

console.warn('MACD signal must be an array');  

}  

if (!Array.isArray(data.technicalIndicators.macd.histogram)) {  

console.warn('MACD histogram must be an array');  

}  

}  

// Validate Bollinger Bands structure  
if (data.technicalIndicators.bollingerBands && typeof data.technicalIndicators.bollingerBands === 'object') {  

if (!Array.isArray(data.technicalIndicators.bollingerBands.upper)) {  

console.warn('Bollinger Bands upper must be an array');  

}  

if (!Array.isArray(data.technicalIndicators.bollingerBands.middle)) {  

console.warn('Bollinger Bands middle must be an array');  

}  

if (!Array.isArray(data.technicalIndicators.bollingerBands.lower)) {  

console.warn('Bollinger Bands lower must be an array');  

}  

}  

// Validate moving averages structure  
if (data.technicalIndicators.movingAverages && typeof data.technicalIndicators.movingAverages === 'object') {  

Object.entries(data.technicalIndicators.movingAverages).forEach(([period, values]) => {  

if (!Array.isArray(values)) {  

console.warn(`Moving average for period ${period} must be an array`);  

}  

});  

}  

// Validate support/resistance levels structure  
if (data.supportResistanceLevels && typeof data.supportResistanceLevels === 'object') {  

if (!Array.isArray(data.supportResistanceLevels.support)) {  

console.warn('Support levels must be an array');  

}  

if (!Array.isArray(data.supportResistanceLevels.resistance)) {  

console.warn('Resistance levels must be an array');  

}  

// Validate each level has required fields  
[...(data.supportResistanceLevels.support || []), ...(data.supportResistanceLevels.resistance || [])].forEach((level, index) => {  

if (!level.price || typeof level.price !== 'number') {  

console.warn(`Level at index ${index} missing valid price`);  

}  

if (!level.strength || typeof level.strength !== 'number') {  

console.warn(`Level at index ${index} missing valid strength`);  

}  

});  

}  

// Validate pattern recognition results structure  
if (data.patternRecognition && typeof data.patternRecognition === 'object') {  

if (!Array.isArray(data.patternRecognition.patterns)) {  

console.warn('Pattern recognition patterns must be an array');  

} else {  

// Validate each pattern has required fields  
data.patternRecognition.patterns.forEach((pattern, index) => {  

if (!pattern.name || typeof pattern.name !== 'string') {  

console.warn(`Pattern at index ${index} missing valid name`);  

}  

if (!pattern.confidence || typeof pattern.confidence !== 'number') {  

console.warn(`Pattern at index ${index} missing valid confidence`);  

}  

if (!pattern.direction || !['bullish', 'bearish'].includes(pattern.direction)) {  

console.warn(`Pattern at index ${index} missing valid direction`);  

}  

});  

}  

// Validate trendline analysis structure  
if (data.trendlineAnalysis && typeof data.trendlineAnalysis === 'object') {  

if (!Array.isArray(data.trendlineAnalysis.trendlines)) {  

console.warn('Trendline analysis trendlines must be an array');  

} else {  

// Validate each trendline has required fields  
data.trendlineAnalysis.trendlines.forEach((trendline, index) => {  

if (!trendline.type || !['support', 'resistance'].includes(trendline.type)) {  

console.warn(`Trendline at index ${index} missing valid type`);  

}  

if (!trendline.slope || typeof trendline.slope !== 'number') {  

console.warn(`Trendline at index ${index} missing valid slope`);  

}  

if (!trendline.intercept || typeof trendline.intercept !== 'number') { 

console.warn(`Trendline at index ${index} missing valid intercept`);

}

});

}

}

// Validate order book structure

if (data.orderBook && typeof data.orderBook === 'object') {

if (!Array.isArray(data.orderBook.bids)) {

console.warn('Order book bids must be an array');

}

if (!Array.isArray(data.orderBook.asks)) {

console.warn('Order book asks must be an array');

}

// Validate each bid/ask has required fields

[...(data.orderBook.bids || []), ...(data.orderBook.asks || [])].forEach((order, index) => {

if (!order.price || typeof order.price !== 'number') {

console.warn(`Order at index ${index} missing valid price`);

}

if (!order.quantity || typeof order.quantity !== 'number') {

console.warn(`Order at index ${index} missing valid quantity`);

}

});

}

// Validate recent trades structure

if (data.recentTrades && Array.isArray(data.recentTrades)) {

data.recentTrades.forEach((trade, index) => {

if (!trade.price || typeof trade.price !== 'number') {

console.warn(`Trade at index ${index} missing valid price`);

}

if (!trade.quantity || typeof trade.quantity !== 'number') {

console.warn(`Trade at index ${index} missing valid quantity`);

}

if (!trade.time || typeof trade.time !== 'number') {

console.warn(`Trade at index ${index} missing valid time`);

}

});

}

// Validate market depth structure

if (data.marketDepth && typeof data.marketDepth === 'object') {

if (!Array.isArray(data.marketDepth.bids)) {

console.warn('Market depth bids must be an array');

}

if (!Array.isArray(data.marketDepth.asks)) {

console.warn('Market depth asks must be an array');

}

}

// Validate funding rate structure

if (data.fundingRate && typeof data.fundingRate === 'object') {

if (typeof data.fundingRate.currentRate !== 'number') {

console.warn('Funding rate currentRate must be a number');

}

if (typeof data.fundingRate.nextFundingTime !== 'number') {

console.warn('Funding rate nextFundingTime must be a number');

}

}

// Validate open interest structure

if (data.openInterest && typeof data.openInterest === 'object') {

if (typeof data.openInterest.total !== 'number') {

console.warn('Open interest total must be a number');

}

if (!Array.isArray(data.openInterest.history)) {

console.warn('Open interest history must be an array');

}

}

// Validate liquidation levels structure

if (data.liquidationLevels && typeof data.liquidationLevels === 'object') {

if (!Array.isArray(data.liquidationLevels.longPositions)) {

console.warn('Liquidation levels longPositions must be an array');

}

if (!Array.isArray(data.liquidationLevels.shortPositions)) {

console.warn('Liquidation levels shortPositions must be an array');

}

}

// Validate market indicators structure

if (data.marketIndicators && typeof data.marketIndicators === 'object') {

const requiredIndicators = ['volatility', 'momentum', 'volumeProfile'];

requiredIndicators.forEach(indicator => {

if (!(indicator in data.marketIndicators)) {

console.warn(`Missing required indicator: ${indicator}`);

}

});

}

// Validate market sentiment structure

if (data.marketSentiment && typeof data.marketSentiment === 'object') {

const requiredFields = ['overallSentiment', 'bullishPercentage', 'bearishPercentage'];

requiredFields.forEach(field => {

if (!(field in data.marketSentiment)) {

console.warn(`Missing required sentiment field: ${field}`);

}

});

}

// Validate news articles structure

if (data.marketNews && Array.isArray(data.marketNews)) {

data.marketNews.forEach((article, index) => {

const requiredFields = ['title', 'url', 'publishedAt'];

requiredFields.forEach(field => {

if (!(field in article)) {

console.warn(`News article at index ${index} missing required field: ${field}`);

}

});

});

}

// Validate events structure

if (data.marketEvents && Array.isArray(data.marketEvents)) {

data.marketEvents.forEach((event, index) => {

const requiredFields = ['name', 'date', 'impact'];

requiredFields.forEach(field => {

if (!(field in event)) {

console.warn(`Event at index ${index} missing required field: ${field}`);

}

});

});

}

// Validate announcements structure

if (data.marketAnnouncements && Array.isArray(data.marketAnnouncements)) {

data.marketAnnouncements.forEach((announcement, index) => {

const requiredFields = ['title', 'content', 'date'];

requiredFields.forEach(field => {

if (!(field in announcement)) {

console.warn(`Announcement at index ${index} missing required field: ${field}`);

}

});

});

}

// Validate warnings structure

if (data.marketWarnings && Array.isArray(data.marketWarnings)) {

data.marketWarnings.forEach((warning, index) => {

const requiredFields = ['type', 'message', 'severity'];

requiredFields.forEach(field => {

if (!(field in warning)) {

console.warn(`Warning at index ${index} missing required field: ${field}`);

}

});

});

}

// Validate errors structure

if (data.marketErrors && Array.isArray(data.marketErrors)) {

data.marketErrors.forEach((error, index) => {

const requiredFields = ['code', 'message'];

requiredFields.forEach(field => {

if (!(field in error)) {

console.warn(`Error at index ${index} missing required field: ${field}`);

}

});

});

}

// Validate logs structure

if (data.marketLogs && Array.isArray(data.marketLogs)) {

data.marketLogs.forEach((log, index) => {

const requiredFields = ['timestamp', 'level', 'message'];

requiredFields.forEach(field => {

if (!(field in log)) {

console.warn(`Log entry at index ${index} missing required field: ${field}`);

}

});

});

}

// Validate metrics structure

if (data.marketMetrics && typeof data.marketMetrics === 'object') {

const requiredMetrics = ['marketCap', 'volume24h', 'circulatingSupply'];

requiredMetrics.forEach(metric => {

if (!(metric in data.marketMetrics)) {

console.warn(`Missing required metric: ${metric}`);

}

});

}

// Validate statistics structure

if (data.marketStatistics && typeof data.marketStatistics === 'object') {

const requiredStats = ['allTimeHigh', 'allTimeLow', 'averageVolume'];

requiredStats.forEach(stat => {

if (!(stat in data.marketStatistics)) {

console.warn(`Missing required statistic: ${stat}`);

}

});

}

// Validate summary structure

if (data.marketSummary && typeof data.marketSummary === 'object') {

const requiredFields = ['currentPrice', 'change24h', 'volume24h'];

requiredFields.forEach(field => {

if (!(field in data.marketSummary)) {

console.warn(`Missing required summary field: ${field}`);

}

});

}

// Validate overview structure

if (data.marketOverview && typeof data.marketOverview === 'object') {

const requiredFields = ['rank', 'marketCap', 'volume24h'];

requiredFields.forEach(field => {

if (!(field in data.marketOverview)) {

console.warn(`Missing required overview field: ${field}`);

}

});

}

// Validate details structure

if (data.marketDetails && typeof data.marketDetails === 'object') {

const requiredFields = ['description', 'websiteUrl', 'whitepaperUrl'];

requiredFields.forEach(field => {

if (!(field in data.marketDetails)) {

console.warn(`Missing required detail field: ${field}`);

}

});

}

// Validate information structure

if (data.marketInformation && typeof data.marketInformation === 'object') {

const requiredFields = ['name', symbol, category];

requiredFields.forEach(field => {

if (!(field in data.marketInformation)) {

console.warn(`Missing required information field: ${field}`);

}

});

}

// Validate status structure

if (data.marketStatus && typeof data.marketStatus === 'object') {

const requiredFields = ['tradingEnabled', depositEnabled, withdrawalEnabled];

requiredFields.forEach(field => {

if (!(field in data.marketStatus)) {

console.warn(`Missing required status field: ${field}`);

}

});

}
}
/**
* Format price value for display
* @param {number|string} price - Price value to format
* @returns {string}
*/
function formatPrice(price) {


const numPrice = parseFloat(price);

return Number(numPrice).toLocaleString('en-US', {


minimumFractionDigits: Math.min(getPriceDecimals(numPrice), getMaxDecimals(numPrice)),


maximumFractionDigits: Math.min(getPriceDecimals(numPrice), getMaxDecimals(numPrice))


});
}
/**
* Determine appropriate decimal places based on price magnitude
* @param {number} price - Price value to analyze
* @returns {number}
*/
function getPriceDecimals(price)


{
return Math.max(2, Math.abs(Math.round(Math.log10(Math.abs(price)))) + 2);


}
/**
* Get maximum decimal places for a given price value based on trading pair conventions.
* This function ensures that prices are displayed with appropriate precision.
* @param {number|string} price - Price value to analyze.
* @returns {number}
*/
function getMaxDecimals(price)


{
return Math.min(8, Math.max(2, Math.abs(Math.round(Math.log10(Math.abs(parseFloat(price)))) + 2)));
}
/**
* Format volume value for display with appropriate unit suffixes.
* This function converts large volume numbers into human-readable format.
* @param {number|string} volume - Volume value to format.
* @returns {string}
*/
function formatVolume(volume)


{
const numVolume = parseFloat(volume);

const suffixes = ['', K, M, B, T];

let suffixIndex = Math.floor(Math.log10(Math.abs(numVolume)) / Math.log10(1000));

return `${Number(numVolume / Math.pow(1000, suffixIndex)).toFixed(2)}${suffixes[suffixIndex]}`;


}
/**
* Initialize the dashboard when DOM is ready.
* This function sets up the real-time update manager and registers event listeners.
*/
document.addEventListener(DOMContentLoaded, () => {


try {


window.dashboardManager = new DashboardUpdateManager();


window.dashboardManager.on(marketData, updateMarketData);


window.dashboardManager.on(signal, handleSignal);


window.dashboardManager.on(alert, handleAlert);


window.dashboardManager.on(status, handleStatus);


window.dashboardManager.on(error, handleError);


window.dashboardManager.on(connection, handleConnectionStatus);


updateConnectionStatusUI();


showNotification(Dashboard initialized successfully);


} catch(error)


{


showNotification(Failed to initialize dashboard:, error.message);


}
});
/**
* Handle incoming trading signals from the server.
* This function processes signal events and updates the UI accordingly.
* @param {Object} signal - Signal object containing trading signal details.
*/
function handleSignal(signal)


{
try


{
const container=document.getElementById(trading-signals-container);


container.innerHTML=``;


const signalCard=document.createElement(div);


signalCard.className=`signal-card signal-${signal.direction.toLowerCase()}`;


signalCard.innerHTML=`


<div class="signal-header">


<span class="signal-symbol">${signal.symbol}</span>


<span class="signal-direction">${signal.direction}</span>


</div>


<div class="signal-details">


<div class="signal-price">Entry: $${formatPrice(signal.price)}</div>


<div class="signal-confidence">Confidence: ${signal.confidence.toFixed(2)}%</div>


<div class="signal-timeframe">Timeframe: ${signal.timeframe}</div>


<div class="signal-timestamp">${new Date(signal.timestamp).toLocaleString()}</div>


</div>


<div class="signal-analysis">


<p>${signal.reasoning}</p>


</div>


<div class="signal-actions">


<button onclick="executeTrade(${JSON.stringify(signal).replace(/"/g,'&quot;')})" class="btn-execute">Execute Trade</button>


<button onclick="dismissSignal($