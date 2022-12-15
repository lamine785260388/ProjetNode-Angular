const { Balance, Agence } = require('../../db/sequelize')
  
module.exports = (app) => {
  app.put('/api/updateBalance/:id', (req, res) => {
    const id = req.params.id
    Agence.findByPk(id).then(balance => {
      if(balance===null){
        const message='l_agence que vous voulez modifier son solde n_existe pas'
        return res.status(404).json({message})
      }
      Balance.update(req.body, {
        where: { AGENCEId: id }
      })
      .then(_=>{
        message=`votre nouvelle solde a été modifié avec succés`
        Balance.findByPk(id).then(balance=>{
          return res.status(201).json({message,data:balance})
        })
       
      })
    })
    .catch(error=>{
      message='le serveur n_est pas disponible veuillez ressayer apres'
      res.status(501).json({message,data:error})
    })
  })
}