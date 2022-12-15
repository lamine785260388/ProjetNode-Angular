const { User } = require('../db/sequelize')
const bcrypt = require('bcrypt')
  const jwt=require('jsonwebtoken')
const privatekey = require('../auth/private_key')
module.exports = (app) => {
  app.post('/api/login', (req, res) => {
  
    User.findOne({ where: { username: req.body.username } }).then(user => {
        if(!user){
        const message='l_utilisateur demande n_existe pas'
        return res.status(404).json({message})
        }

         
      bcrypt.compare(req.body.password, user.password).then(isPasswordValid => {
        if(!isPasswordValid) {
          const message = `Le mot de passe est incorrecte`;
          return res.status(404).json({ message })
        }
        //JWT
     const token=jwt.sign(
        {userId:user.id},
        privatekey,
        {expiresIn:"24h"}
     )
      iduser=user.id
      console.log(iduser)
        const message = `L'utilisateur a été connecté avec succès2`;
       
       const islogin='true';
        return res.json({ message, data: user,token,islogin })
      })
    })
     .catch(error=>{
        const message='l_utilisateur n_a pas pu etre connecte avec succes'
        return res.json({message,data:error})
     })
  })
}