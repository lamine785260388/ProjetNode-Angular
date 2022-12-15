module.exports = (sequelize, DataTypes) => {
    return sequelize.define('PAYS', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      code_iso2_pays:{
      type:DataTypes.STRING
      },
      nom_pays: {
        type: DataTypes.STRING,

      }
      
     
       

    })
  }