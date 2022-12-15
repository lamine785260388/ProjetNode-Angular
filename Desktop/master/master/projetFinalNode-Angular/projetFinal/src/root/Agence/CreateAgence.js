const { Agence } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertAgence', (req, res) => {

  
   Agence.create({
    code_agence:req.body.code_agence,
    nom_agence:req.body.nom_agence,
    statut_agence:req.body.statut_agence
  

    })
    return res.json('insertion Agence success')
    
   
   
  })
}