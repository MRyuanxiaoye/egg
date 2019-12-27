const merge = require('webpack-merge');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  devtool: 'inline-source-map',
  devServer: {
    contentBase: './dist',
    compress: true,
    host: '0.0.0.0',
    port: 9000,
    proxy: {
      '/api': 'http://127.0.0.1:8080',
    }
  }
});