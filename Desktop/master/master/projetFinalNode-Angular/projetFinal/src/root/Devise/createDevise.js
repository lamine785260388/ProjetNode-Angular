const { Devise } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertDevise', (req, res) => {

  
   Devise.create({
    code_iso3:req.body.code_iso3,
    nom_devise:req.body.nom_devise,
    role:req.body.role,
    symbole:req.body.symbole
  

    })
    return res.json('issertion Devise success')
    
   
   
  })
}