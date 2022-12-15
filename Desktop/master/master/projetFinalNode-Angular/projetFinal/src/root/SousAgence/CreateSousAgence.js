const { SousAgence } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertSousAgence', (req, res) => {

  
   SousAgence.create({
    code_sous_agence:req.body.code_sous_agence,
    nom_sous_agence:req.body.nom_sous_agence,
    addresse_sous_agence:req.body.addresse_sous_agence,
    city_sous_agence:req.body.city_sous_agence,
    country_sous_agence:req.body.country_sous_agence,
    phone_sous_agence:req.body.phone_sous_agence,
    email_sous_agence:req.body.email_sous_agence,
    AGENCEId:+req.body.AGENCEId
  

    })
    return res.json('insertion SousAgence success')
    
   
   
  })
}