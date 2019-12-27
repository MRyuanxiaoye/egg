import axios from 'axios'

axios.get('/api/aaa')
  .then((re) => {
    console.log(re);
  })