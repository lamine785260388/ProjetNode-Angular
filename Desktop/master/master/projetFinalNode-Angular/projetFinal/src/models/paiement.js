module.exports = (sequelize, DataTypes) => {
    return sequelize.define('PAIEMENT', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
   
      numero_piece: {
        type: DataTypes.STRING
      },
      nom_recepteur: {
        type: DataTypes.STRING,
      },
      TRANSACTIONId:{
          type:DataTypes.INTEGER,
      }
      
       

    })
  }