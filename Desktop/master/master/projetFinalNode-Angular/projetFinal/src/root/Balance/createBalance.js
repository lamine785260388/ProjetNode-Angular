const { Balance } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertBalance', (req, res) => {

  
   Balance.create({
    montant:+req.body.montant,
    AGENCEId:+req.body.AGENCEId

    })
    
    return res.json('insertion Balance success')
    
   
   
  })
}