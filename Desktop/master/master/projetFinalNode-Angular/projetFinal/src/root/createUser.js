const { User } = require('../db/sequelize')
const bcrypt=require('bcrypt')
  
module.exports = (app) => {
  app.post('/api/Inscrire', (req, res) => {
    

  
    bcrypt.hash(req.body.password,10)
    .then(hash=>User.create({
      username:req.body.username,
      isAdmin:'false',
      password:hash,
      SOUSAGENCEId:req.body.SOUSAGENCEId

    })
    .then(result=>{
      if(result!=null){
      erreur=false
       message="Utilisateur Inserer avec Succe"
       return res.json({erreur,message})
      }
      else{
        message='Impossible d_enregistrer l_utilisateur'
        res.json({message})
      }

    })
    .catch(erreur=>{
      message='Impossible d_enregistrer l_utilisateur'
      res.json({message,data:erreur})

    })
    )
    
    
   
   
  })
}