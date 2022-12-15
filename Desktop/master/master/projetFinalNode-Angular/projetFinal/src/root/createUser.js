const { User } = require('../db/sequelize')
const bcrypt=require('bcrypt')
  
module.exports = (app) => {
  app.post('/api/Inscrire', (req, res) => {

  
    bcrypt.hash(req.body.password,10)
    .then(hash=>User.create({
      username:req.body.username,
      isAdmin:'false',
      password:hash
    }))
    return res.json('inscription success')
    
   
   
  })
}